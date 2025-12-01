GeoExplicit-SFM 学术展示讲稿

场合：HKUST(GZ) 学术汇报 / 课程展示
时长：约 10-12 分钟
演讲者：[您的名字]

[Slide 1: Title Slide]

(动作：自信地站定，微笑，环视听众)

Speaker:
Good morning/afternoon, Professors and fellow students. I am [您的名字] from the Smart Urban Planning Group.

Today, I am honored to present our research titled: "GeoExplicit-SFM: Geographically Explicit Social Force Model with Diffusion Policy for Urban Trajectory Generation."

In this work, we explore how to bridge the gap between traditional physics-based simulations and modern generative AI to solve the data scarcity problem in urban computing.

[Slide 2: Research Context]

(动作：切换 PPT，手势指向屏幕上的三个图标)

Speaker:
Let's start with the motivation. As we know, human trajectory data is the lifeblood of understanding city dynamics. However, obtaining high-quality, individual-level real-world data is extremely difficult due to privacy concerns and sensor costs.

Yet, this data is crucial for downstream tasks such as:

Traffic Planning: Understanding commuting flows.

Epidemic Control: Simulating virus transmission.

Urban Design: Optimizing facility layouts based on pedestrian hotspots.

So, the question is: How can we generate realistic urban trajectories without violating privacy?

[Slide 3: Current Limitations]

(动作：切换 PPT，对比左右两栏)

Speaker:
Existing methods generally fall into two categories, each with its own limitations:

On the left, we have Traditional Physics Models, like the Social Force Model (SFM). They are physically consistent and interpretable. However, they often ignore complex geographic constraints—like walls or buildings—and require manually designing potential fields, which is not scalable for a real city.

On the right, we have Deep Learning Methods, like GANs or Transformers. While powerful, they act as "Black Boxes" and, ironically, they require massive amounts of training data to begin with.

Our Goal is to combine the best of both worlds: creating a framework that is Geographically Explicit, Physics-Driven, yet capable of Policy Learning.

[Slide 4: Methodology Framework]

(动作：切换 PPT，简单概括流程)

Speaker:
To achieve this, we propose a four-phase pipeline:

Phase 1 (Sinks): Identifying where agents want to go based on real GIS data.

Phase 2 (Fields): Constructing a gravity-based potential field to guide them.

Phase 3 (Sim): Running a Langevin Dynamics simulation to generate high-fidelity data.

Phase 4 (Learn): Distilling this physical behavior into a lightweight Diffusion Policy for rapid inference.

Let's dive into the details.

[Slide 5: Phase 1 - Sink Identification]

(动作：切换 PPT，指向右侧的 GIS 地图)

Speaker:
First, we ground our simulation in real commuting patterns.

We use **Census LODES Origin-Destination flow data**, which captures where people actually travel for work. We apply **Pareto filtering** to focus on high-flow destinations—the top 80% of commuting volume.

Then, we use **DBSCAN spatial clustering** with Haversine distance to merge nearby high-flow zones into coherent destination clusters.

In Wayne County—the Detroit metro area—this yields **35 major Sinks**, each representing thousands of daily commuters.

[Slide 6: Phase 2 - Navigation Field]

(动作：切换 PPT，展示右侧的势场图)

Speaker:
With destinations identified, how do we guide agents toward them?

We compute a **distance-based navigation field**. For each walkable pixel, we calculate the direction pointing toward the nearest sink along the road network.

This creates a smooth vector field—as shown in the visualization—where every location has a clear "pull" direction. Unlike Eikonal shortest-path methods, our field naturally handles multiple competing destinations weighted by their attractiveness.

[Slide 7: Mathematical Formulation]

(动作：切换 PPT，简要解释公式，不要逐字读符号)

Speaker:
Mathematically, we formulate the potential $\phi(x)$ as the sum of inverse distances to all $N$ sinks.
The parameter $w_i$ represents the weight or "attractiveness" of a destination, and $\alpha$ controls the decay rate.

By differentiating this potential field, we obtain a gradient vector that naturally guides the agent towards high-utility areas while maintaining smooth trajectories.

[Slide 8: Phase 3 - Langevin Dynamics]

(动作：切换 PPT，强调中间的公式结构)

Speaker:
With the field constructed, we move to Phase 3: The Simulation.
We employ Overdamped Langevin Dynamics.

The movement equation consists of three parts:

Inertia: Controlled by momentum $\mu$, ensuring smooth motion.

Potential Guidance: The gradient we just calculated, pulling the agent towards goals.

Stochastic Noise: A Gaussian noise term $\sigma$, simulating human variability and exploration.

We calibrated these parameters—setting momentum to 0.7 and noise scale to 0.08—to balance smooth motion with realistic human variability.

[Slide 9: Geographic Constraints]

(动作：切换 PPT，强调右下角的 Road Mask 图示)

Speaker:
Physics alone isn't enough—we need strict geographic consistency.

We enforce a **walkable mask** derived from real road shapefiles. Agents are confined to valid road pixels only.

When an agent attempts an invalid move, we apply **boundary recovery**: it's gently pushed back onto the nearest road.

With vectorized NumPy operations, we simulate **10,000 agents over 10,000 steps** in under a minute on a standard workstation.

[Slide 10: Results]

(动作：切换 PPT，指向数据)

Speaker:
The simulation results are highly promising.

We achieved **100% road adherence**—agents never walk through buildings or off-road areas. The average speed remains stable at 0.75 pixels per step, with smooth directional changes.

We generated **10,000 agent trajectories over 10,000 time steps**—that's 100 million data points—serving as high-fidelity training data for the next phase.

[Slide 11: Phase 4 - Policy Learning]

(动作：切换 PPT，对比左右两个方框)

Speaker:
You might ask: If the simulation works well, why do we need Phase 4?

The problem is that the Physics Simulation is computationally heavy. It requires loading large navigation fields and performing gradient lookups at every step—not ideal for real-time or edge deployment.

Therefore, we use **Diffusion Policy Distillation**. The simulator acts as a "Teacher," generating expert demonstrations. We train a lightweight neural network "Student" that learns to imitate this behavior.

The result? A model **under 5MB** that runs in milliseconds per step, without needing any field arrays at inference time.

[Slide 12: Model Architecture]

(动作：切换 PPT，简述网络结构)

Speaker:
We adopted a **1D-UNet** architecture with a **DDPM denoising scheduler**.

The model takes a 2-frame history—position and velocity—as conditioning input. Through iterative denoising, it transforms random noise into a coherent **8-step future action sequence**.

Crucially, diffusion models excel at capturing **multi-modal distributions**. Unlike regression models that output a single "average" path, our model can sample diverse yet realistic trajectories—essential for simulating the variability in human movement.

[Slide 13: Conclusion]

(动作：切换 PPT，总结核心贡献)

Speaker:
To summarize, our work makes three key contributions:

Geographic Explicit: We integrated real GIS data directly into the physics model.

Physics-Data Fusion: We combined the reliability of Langevin dynamics with the flexibility of Generative AI.

End-to-End Pipeline: We established a complete workflow from raw shapefiles to a deployable behavioral policy.

[Slide 14: Future Directions]

(动作：切换 PPT，展望未来)

Speaker:
Looking forward, there are several exciting directions:

We plan to incorporate Real Origin-Destination (OD) data to calibrate the sink weights.

We aim to extend this to Multi-Modal Transport, including vehicles.

And finally, we are exploring the use of LLM Agents to drive high-level strategic decision-making for the agents.

[Slide 15: Q&A]

(动作：切换 PPT，微笑致意)

Speaker:
That concludes my presentation. Thank you very much for your attention.
I am now happy to take any questions.