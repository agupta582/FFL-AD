## Is Performance Fairness Achievable in Presence of Attackers under Federated Learning? 
##### **To be appeared in ECAI 2023**

###### **Abstract**:

In the last few years, Federated Learning (FL) has received extensive attention from the research community because of its capability for privacy-preserving, collaborative learning from heterogeneous data sources. Most FL studies focus on either average performance improvement or the robustness to attacks, while some attempt to solve both jointly. However, the performance disparities across clients in the presence of attackers have largely been unexplored. In this work, we propose a novel Fair Federated Learning scheme with Attacker Detection capability (abbreviated as FFL+AD) to minimize performance discrepancies across benign participants. FFL+AD enables the server to identify attackers and learn their malign intent (e.g., targeted label) by investigating suspected models via top performers. This two-step detection method helps reduce false positives. Later, we introduce fairness by regularizing the benign clients’ local objectives with a variable boosting parameter that gives more emphasis on low performers in optimization. Under standard assumptions, FFL+AD exhibits a convergence rate similar to FedAvg. Experimental results show that our scheme builds a more fair and more robust model, under label-flipping and backdoor attackers, compared to prior schemes. FFL+AD achieves competitive
accuracy even when 40% of the clients are attackers.


![FFL+AD Overview](https://github.com/agupta582/FFL-AD/assets/67119520/7f1480b7-cd96-42da-a2b2-4a450512ef3f)



#### **Execute a scenario via terminal**

python main.py -dataset MNIST -gpu no -flRounds 30 -numClients 50 -clientEpochs 4 -atk_backdoor_list 1,3,5,7 -atk_label_flip_unidir_list 2,4,6,8

#### **See "getArguments.py" for generating more scenarios** 
