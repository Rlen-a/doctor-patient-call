#!/usr/bin/env python3
"""
Сервер телемедицинской видеосвязи (сигнальный сервер для WebRTC).

Что делает:
  * при запуске создаёт комнату для конкретного врача и конкретного пациента;
  * генерирует два ОДНОРАЗОВЫХ токена (врача и пациента) и печатает их в терминал;
  * принимает токен на /api/room/join, "сжигает" его и выдаёт сессионный ключ,
    привязанный к роли, имени и IP-адресу компьютера;
  * пропускает обмен SDP (offer/answer) только между этими двумя участниками:
    offer может отправить только пациент, answer — только врач.
Сами аудио/видео идут напрямую между врачом и пациентом (WebRTC, peer-to-peer).

Запуск:  python server.py --doctor "Иванов И.И." --patient "Петров П.П."
"""
import argparse
import asyncio
import json
import secrets
import socket
import time
import uuid

from aiohttp import web


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # пакеты не отправляются, нужен лишь выбор интерфейса
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def fail(cls, msg: str):
    raise cls(text=json.dumps({"error": msg}, ensure_ascii=False),
              content_type="application/json")


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def auth(request: web.Request, role: str) -> dict:
    st = request.app["state"]
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        fail(web.HTTPUnauthorized, "нет сессионного ключа")
    sess = st["sessions"].get(header[7:])
    if sess is None:
        fail(web.HTTPUnauthorized, "неизвестная сессия")
    if sess["ip"] != request.remote:
        fail(web.HTTPForbidden, "сессия принадлежит другому компьютеру")
    if sess["role"] != role:
        fail(web.HTTPForbidden, f"операция доступна только роли '{role}'")
    return sess


async def read_sdp(request: web.Request, expected_type: str) -> dict:
    try:
        data = await request.json()
        sdp, typ = data["sdp"], data["type"]
    except Exception:
        fail(web.HTTPBadRequest, "ожидается JSON {sdp, type}")
    if typ != expected_type or not isinstance(sdp, str):
        fail(web.HTTPBadRequest, f"ожидается type='{expected_type}'")
    return {"sdp": sdp, "type": typ}


async def wait_for(event: asyncio.Event, timeout: float, what: str):
    try:
        await asyncio.wait_for(event.wait(), timeout)
    except asyncio.TimeoutError:
        fail(web.HTTPRequestTimeout, f"время ожидания истекло: {what}")


# ---------------------------------------------------------------- ручки API

async def join(request: web.Request):
    """POST /api/room/join {"token": "..."} -> {"session", "role", "name", "room_id"}"""
    st = request.app["state"]
    try:
        token = str((await request.json()).get("token", ""))
    except Exception:
        fail(web.HTTPBadRequest, "ожидается JSON {token}")
    rec = st["tokens"].get(token)
    if rec is None:
        log(f"отказ: неверный токен от {request.remote}")
        fail(web.HTTPUnauthorized, "неверный токен")
    if rec["used"]:
        log(f"отказ: повторное использование токена ({rec['role']}) от {request.remote}")
        fail(web.HTTPConflict, "токен уже использован")
    if time.time() > rec["expires"]:
        fail(web.HTTPUnauthorized, "срок действия токена истёк")

    rec["used"] = True  # одноразовый: сжигаем сразу
    session = secrets.token_urlsafe(32)
    st["sessions"][session] = {"role": rec["role"], "name": rec["name"], "ip": request.remote}
    st["joined"].add(rec["role"])
    log(f"{rec['role']} '{rec['name']}' вошёл в комнату {st['room_id']} с {request.remote}")
    if st["joined"] == {"doctor", "patient"}:
        st["both"].set()
        log("врач и пациент на месте, обмен SDP разрешён")
    return web.json_response({"session": session, "role": rec["role"],
                              "name": rec["name"], "room_id": st["room_id"]})


async def post_offer(request: web.Request):
    """POST /api/signal/offer — только пациент"""
    st = request.app["state"]
    auth(request, "patient")
    offer = await read_sdp(request, "offer")
    await wait_for(st["both"], st["wait"], "ожидание второго участника")
    if st["offer"] is not None:
        fail(web.HTTPConflict, "offer уже отправлен")
    st["offer"] = offer
    st["offer_ev"].set()
    log("получен offer от пациента")
    return web.json_response({"ok": True})


async def get_offer(request: web.Request):
    """GET /api/signal/offer — только врач (ждёт появления offer)"""
    st = request.app["state"]
    auth(request, "doctor")
    await wait_for(st["offer_ev"], st["wait"], "ожидание offer")
    log("offer выдан врачу")
    return web.json_response(st["offer"])


async def post_answer(request: web.Request):
    """POST /api/signal/answer — только врач"""
    st = request.app["state"]
    auth(request, "doctor")
    answer = await read_sdp(request, "answer")
    if st["offer"] is None:
        fail(web.HTTPConflict, "offer ещё не получен")
    if st["answer"] is not None:
        fail(web.HTTPConflict, "answer уже отправлен")
    st["answer"] = answer
    st["answer_ev"].set()
    log("получен answer от врача")
    return web.json_response({"ok": True})


async def get_answer(request: web.Request):
    """GET /api/signal/answer — только пациент (ждёт появления answer)"""
    st = request.app["state"]
    auth(request, "patient")
    await wait_for(st["answer_ev"], st["wait"], "ожидание answer")
    log("answer выдан пациенту — соединение устанавливается")
    return web.json_response(st["answer"])


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Сервер видеосвязи врач-пациент")
    p.add_argument("--doctor", required=True, help="имя врача")
    p.add_argument("--patient", required=True, help="имя пациента")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ttl", type=int, default=900, help="срок жизни токена, сек (по умолчанию 900)")
    p.add_argument("--wait", type=int, default=300, help="макс. ожидание партнёра, сек")
    args = p.parse_args()

    room_id = uuid.uuid4().hex[:8]
    tokens, issued = {}, {}
    for role, name in (("doctor", args.doctor), ("patient", args.patient)):
        t = secrets.token_urlsafe(24)
        tokens[t] = {"role": role, "name": name, "used": False,
                     "expires": time.time() + args.ttl}
        issued[role] = t

    state = {
        "room_id": room_id, "tokens": tokens, "sessions": {}, "joined": set(),
        "both": asyncio.Event(), "offer": None, "offer_ev": asyncio.Event(),
        "answer": None, "answer_ev": asyncio.Event(), "wait": args.wait,
    }
    app = web.Application()
    app["state"] = state
    app.add_routes([
        web.post("/api/room/join", join),
        web.post("/api/signal/offer", post_offer),
        web.get("/api/signal/offer", get_offer),
        web.post("/api/signal/answer", post_answer),
        web.get("/api/signal/answer", get_answer),
    ])

    ip = lan_ip()
    print("=" * 66)
    print(f" Комната: {room_id}")
    print(f" Адрес сервера для клиентов:  http://{ip}:{args.port}")
    print(f" Токены одноразовые, действуют {args.ttl} сек.")
    print("-" * 66)
    print(f" ВРАЧ    ({args.doctor}):\n   {issued['doctor']}")
    print(f" ПАЦИЕНТ ({args.patient}):\n   {issued['patient']}")
    print("=" * 66)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
