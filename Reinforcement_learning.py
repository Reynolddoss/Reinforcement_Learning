# creating the Q matrix for optimised path to reach the goal 

import numpy as np
import random 

#initializing the q matrix and the R matrix 
Q=np.zeros(49).reshape(7,7)
#print(Q)

# creating  Reward matrix R
R=np.zeros(49).reshape(7,7)

for i in range(7):
	for j in range(7):
		R[i][j]=-1

#Assigning Rewards 
R[0,2]=0
R[1,3]=0
R[1,6]=100
R[2,0]=0
R[2,3]=0
R[3,1]=0
R[3,2]=0
R[3,4]=0
R[3,4]=0
R[3,5]=0
R[4,3]=0
R[5,3]=0
R[5,6]=100
R[6,1]=0
R[6,5]=0
R[6,6]=100

#print(R)

def find(state,R,Q,gamma):
	path=[]
	for i in range(7):
		if R[state,i]>=0:
			path.append(i)
	selected_path=random.choice(path)
	
	#print(path)
	new=[]
	for i in range(7):
		if(R[selected_path,i]>=0):
			new.append(Q[selected_path,i])
	Q_max=max(new)
	Q[state,selected_path]=R[state,selected_path]+(gamma*Q_max)
	return (selected_path)



# the learning factor gamma=0.7
gamma=0.7
episode=10000
goal_state=6
for i in range(episode):
	current_state=random.randint(0,6)
	while True:
		next_state=find(current_state,R,Q,gamma)
		current_state=next_state
		if (current_state==goal_state):
			break


# normalize the Q matrix with the highest value and multiply with 100
def Q_normalize(Q):
    max_val=np.max(Q)
    return np.multiply((np.divide(Q,max_val)),100)

print(Q_normalize(Q))
# now the shortest path from the normalised Q matrix 

# Once the Q matrix gets closed enough to a state of convergence, we know our agent has learned 
# the most optimal path to the goal state. Tacing the best sequence of state is as
# simple as following the links the highest values at each state

memory=Q_normalize(Q)
def result_path(start_state,goal_state):
    path=[]
    path.append(start_state)
    while True:
        t=max(memory[start_state])
        c=np.where(memory[start_state]==t)
        x=np.squeeze(c)
        try:
        	temp_path=x[0]
        except (IndexError):
            temp_path=np.max(x)
        start_state=temp_path
        path.append(temp_path)
        if(start_state==goal_state):
             break
    return path
print("The path to reach the goal from the current state is : \n")
print(result_path(2,6))
