import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(color_codes=True)

def read_json():
    with open('data_log_test.json', 'r') as f:
        log=json.load(f)
    return log

if __name__ == "__main__":    
    log =read_json()

    steer = np.array(log["steer"])
    accel = np.array(log["accel"])
    breakk = np.array(log["break"])
    eps_leng = np.array(log["epsiode length"])
    reward = np.array(log["reward"])
    
    

    plt.figure()
    plt.hist(accel)
    plt.title('Acceleration Histogram')
    plt.savefig('accel.png')

    plt.figure()
    plt.hist(breakk)
    plt.title('Break Histogram')
    plt.savefig('breakk.png')

    plt.figure()
    plt.hist(steer)
    plt.title('Steer Histogram')
    plt.savefig('steer.png')
    

    plt.figure()
    plt.plot(reward)
    plt.title('Reward vs Step')
    plt.savefig('reward.png')

    