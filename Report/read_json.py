import json

def read_json()
    with open('data_log_test.json', 'r') as f:
        log=json.load(f)
return log

if __name__ == "__main__":    
    log =read_json()
    
    log["steer"]
    log["accel"]
    log["break"]
    log["epsiode length"]
    log["reward"]