#!/usr/bin/env python3
"""
Клиент ПАЦИЕНТА.
Запуск:  python patient.py --server http://192.168.1.10:8080
Токен пациента (его печатает сервер) вводится в терминале.
Рядом должны лежать call_common.py и face_check.py.
Выход: клавиша 'q' в окне видео или Ctrl+C.
"""
from call_common import RTCSessionDescription, api, client_main


async def negotiate(pc, http, base, headers, add_tracks):
    add_tracks()
    await pc.setLocalDescription(await pc.createOffer())
    await api(http, "POST", f"{base}/api/signal/offer", headers=headers,
              json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    print("Offer отправлен, ждём врача...")
    answer = await api(http, "GET", f"{base}/api/signal/answer", headers=headers)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"],
                                                        type=answer["type"]))


if __name__ == "__main__":
    # "Врач" — как называть собеседника в сообщениях о лице
    client_main("patient", "Врач", "Doctor", negotiate)
