# experiment 1

---
## env
* random goal

## network
```
layer:
   h_1 = 512
   h_2 = 512
```

## state (5+2+360)
* robot location(pos and ang(rad)) 3
* goal and robot dis and ang(rad)  2
* robot vec                        2
* scan                             360

## action
* move angle

## reward
```
reward = -dis/40.0
# 120
reward += round(4.76*(1.05-abs(obs[4]-action[1])),2)

if(info == 'goal' and self.goal_count >= 50):
    reward += 100.
elif(info == 'goal' and self.goal_count < 50):
    reward += 10.
elif(info == 'bump'):
    reward -= 500.
elif(info == 'over range'):
    reward -= 500.
```

## model
1. model_1
    * obstacle 5
    * episodes : 3000
    * step : 2000
    * success rate : 60%
    * train time : 17h

## result
* 只能稍微避障
* 無法避開終點前障礙物
    
