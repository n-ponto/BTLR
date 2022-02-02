import time
import tinytuya
# https://github.com/jasonacox/tinytuya

DEVICE_ID_HERE = '36543161840d8e525100'
IP_ADDRESS_HERE = '10.0.0.238'
LOCAL_KEY_HERE = 'bb05c0afa59e3f17'

if __name__ == "__main__":
    d = tinytuya.OutletDevice(DEVICE_ID_HERE, IP_ADDRESS_HERE, LOCAL_KEY_HERE)
    d.set_version(3.1)
    data = d.status()  
    print(f"data: {data}")
    for _ in range(3):
        time.sleep(1)
        d.turn_off()
        time.sleep(1)
        d.turn_on()
