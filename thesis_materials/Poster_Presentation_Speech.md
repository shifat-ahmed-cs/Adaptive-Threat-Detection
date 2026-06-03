# Adaptive Threat Detection: Full Poster Walkthrough Speech

*Use this speech to guide the audience through your poster from Section 01 to 10.*

### **Introduction & Problem (Sections 01 & 02)**
"Hello, I’m [Name]. Our research, **'Adaptive Threat Detection for Securing Cloud CI/CD Pipelines,'** addresses a critical security gap.
As you can see in the **Abstract (01)**, modern CI/CD pipelines move fast, but traditional security tools are static and slow. They fail against polymorphic malware that changes its appearance.
Our **Research Objective (02)** was to design a self-adaptive neural network that integrates directly into the pipeline, allowing for real-time monitoring and continuous self-learning through threat mutation."

### **Background & Data (Sections 03 & 04)**
"Moving to the **Literature Review (03)**, effectiveness of existing methods like M-CNN and LSTM is limited. As shown in the comparison table, while methods like *Kolosnjaji et al.* used LSTM on system calls, they lacked the multi-modal approach we needed.
For our **Dataset (04)**, we used the Malware Analysis Datasets [21]. We process investigating samples in two ways: as **Grayscale Images** (to see spatial structure) and as **Dynamic API Call Sequences** (to see behavioral patterns)."

### **Our Approach & Methodology (Section 05)**
"This brings us to our core **Methodology (05)**, which represents our unique approach to the solution.
Our strategy is built on three pillars: **Dual-Modal Analysis**, **Real-Time Integration**, and **Continuous Evolution**.

1.  **Source & Initiation:** Code commits trigger our CI Runner.
2.  **Dual-Branch Model:** We send the binary to a **CNN (Spatial)** and an **LSTM (Temporal)** branch simultaneously.
3.  **Feature Fusion:** We merge these insights to get a complete picture.
4.  **Detection & Action:** The system decides if it's 'Benign' (Release to Production) or 'Malicious' (Quarantine/Revoke Credentials).

Crucial to our approach is the **Continuous Learning & Model Evolution** loop at the bottom, where the model trains on mutated data to stay ahead of attackers."

### **Experimental Comparison (Sections 06 & 07)**
"We rigorously tested our architecture choices.
In **Section 06 (Feature Fusion Comparison)**, you can see that **Concatenation** achieved the highest accuracy significantly outperforming Addition and Attention mechanisms.
In **Section 07 (Mutation Strategy Comparison)**, we tested how to best 'harden' the model. The **'Random Insert'** strategy (inserting junk API calls) gave us the highest robustness (92.6%), proving superior to simple appending or replacing."

### **Analysis & Results (Sections 08 & 09)**
"Our **Analysis (08)** visualizes the performance.
*   The **Confusion Matrix** shows very low false positives (only 16 'False Malicious' vs. 6282 'True Malicious').
*   The **ROC Curve** confirms a near-perfect classification capability.
*   The **Activation Heatmaps** on the right show exactly which parts of the malware binary the CNN is focusing on.

In **Section 09 (Result)**, we benchmarked against other systems. Our Dual-Branch method achieved **98.65% Accuracy**, beating standard CNN methods (98.52%) and Graph CNN approaches (92.44%).

**Why is our method superior?**
Unlike traditional **SAST and DAST** tools that rely on static signatures or limited execution paths, our system is far more resilient. Traditional tools fail against **polymorphic malware** which constantly changes its code structure. By combining **visual structure (CNN)** and **temporal behavior (LSTM)**, our model captures the fundamental nature of the threat, not just its surface appearance. This multi-modal approach makes it significantly harder for attackers to evade detection compared to standard single-method scanners."

### **Future Work (Section 10)**
"Finally, for **Future Work (10)**, we plan to expand dataset diversity to include PDF malware and replace our random mutation logic with **Generative Adversarial Networks (GANs)** for even more sophisticated adversarial training.

Thank you! I'm happy to answer any questions."
