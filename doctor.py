#!/usr/bin/env python3
"""
Клиент ВРАЧА.
Запуск:  python doctor.py --server http://192.168.1.10:8080
Токен врача (его печатает сервер) вводится в терминале.
Рядом должны лежать call_common.py и face_check.py.
Выход: клавиша 'q' в окне видео или Ctrl+C.
"""
from call_common import RTCSessionDescription, api, client_main


async def negotiate(pc, http, base, headers, add_tracks):
    print("Ждём offer от пациента...")
    offer = await api(http, "GET", f"{base}/api/signal/offer", headers=headers)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"],
                                                        type=offer["type"]))
    add_tracks()  # у врача треки добавляются после получения offer
    await pc.setLocalDescription(await pc.createAnswer())
    await api(http, "POST", f"{base}/api/signal/answer", headers=headers,
              json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    print("Answer отправлен, устанавливаем соединение...")


if __name__ == "__main__":
    # "Пациент" — как называть собеседника в сообщениях о лице
    client_main("doctor", "Пациент", "Patient", negotiate)
