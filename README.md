# FFL-AD

In this work, we propose a novel Fair Federated Learning scheme with Attacker Detection capability (abbreviated as FFL+AD) to minimize performance discrepancies across benign participants. FFL+AD enables the server to identify attackers and learn their malign intent (e.g., targeted label) by investigating suspected models via top performers. This two-step detection method helps reduce false positives. Later, we introduce fairness by regularizing the benign clients’ local objectives with a variable boosting parameter that gives more emphasis on low performers in optimization. Under standard assumptions, FFL+AD exhibits a convergence rate similar to FedAvg. Experimental results show that our scheme builds a more fair and more robust model, under label-flipping and backdoor attackers, compared to prior schemes. FFL+AD achieves competitive accuracy even when 40% of the clients are attackers.


![FFL+AD Overview](https://github.com/agupta582/FFL-AD/assets/67119520/7f1480b7-cd96-42da-a2b2-4a450512ef3f)
