# Vector Net
This repo works as a personal record of vector net, which serves as part of our trajectory prediction task. The dataset used is INTERACTION by MSC lab of University of California, Berkeley.

The model now is deployed to use information of HD map and all cars involved in the scene to predict the trajectory of the single agent.

# Code Structure:
dataloader_osm.py defines the dataloader of the task. The 

extract_osm.py extract information from osm maps and encode them into embeddings.

feature.py is used to get data from csv files (segmented: frame_n = 40, frame_gap = 10, where I use 10 frames as known information to predict next 30 frames). Ihis code can find the number of shared cars in the 40 frames and use the smallest number as the final number of agents. Then it will take information of the selected agents and stores it as a npy file. The agent feature will be embedded as a 10 element vector where x,y,vx,vy, psi and other valued information can be found. 
The code can only work for DR_CHN_Merging_SZ as it can use int to model agent_id where "P1"

For vector net alone, I tune it with 20 frames as known to predict next 20 frames.

sub_graph.py builds sub_graph like the paper, and then the global_graph.py aggeragate information of sub_graphs to generate global graph.

The whole model is built in vector_net.py. As for now, I use MLP for prediction like the paper, I am working to find a better method to generate better results.

# Undergoing
* Now the vertex number in a vector is strictly limited to 15, which means if the polyline is longger than 15 vertices, then there will be more then one vector represent the polyline (with the same polyline id), if the polyline obtains less than 15 vertices, then it will be filled with zeros. However, there is a way to make the vector as long as it should be. Just need a little more coding. Then the number of agents will not be limited any more. (This part hase been finished, right now we are testing its performance.)
* How to model the self attention properly? There might still be some problem with the model.

# Exeperiment
* The feature.py is only for DR_CHN_Merging_SZ
* The results for this scenario is: train/val: 1.35/1.38 (ade)
* The code is not the final version, now the model is used to predict the final position of the agent, so the out_dim of vector_net.py is 2, which will be supervised by the position of the 40th frame. The average error is about 6. We try this because we find that the final position is hardest to predict and we try to figure out how can this extrctor get the most of infomation out of a HD-map.
* If you want to predict the whole trajectory. change out_dim in the last class of vector_net.py to the double of your expected frame number, and the output shall be taken as the offset from the last known position, which means a little modificatin in main.py as well. 

# Contact
If you have any better idea of extracting infomation from HD-map to help traj prediction, please contact me:

jiefeng@berkeley.edu
