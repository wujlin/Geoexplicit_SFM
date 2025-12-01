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
First, we need to ground our simulation in reality. We don't just randomly place destinations.

We ingest raw GeoPackage data—including schools, offices, and daycares—and transform them from WGS84 coordinates into a raster grid.
Crucially, strictly mapping points to a grid isn't enough. We perform network snapping, ensuring every Point of Interest (POI) is snapped to the nearest road network node. In our case study of Wayne County, we identified 166 valid "Sinks" (destinations) that ensure agents always have a reachable goal.

[Slide 6: Phase 2 - Potential Fields]

(动作：切换 PPT，展示右侧的 3D 势场图)

Speaker:
Once we have the destinations, how do we guide the agents?
Traditional methods often use "Eikonal equations" which simply find the shortest path to the nearest exit. This is unrealistic for city dwellers who have specific preferences.

Instead, we propose a Gravity Potential Field.
As you can see in the visualization, this creates a smooth, continuous landscape where agents feel the "pull" of multiple destinations simultaneously, weighted by the attractiveness of each sink. This eliminates the sharp, robotic turns often seen in Voronoi-based approaches.

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

We calibrated these parameters—setting momentum to 0.85 and noise to 0.05—to mimic realistic walking behaviors.

[Slide 9: Strict Constraints]

(动作：切换 PPT，强调右下角的 Road Mask 图示)

Speaker:
However, physics alone isn't enough; we need geographic consistency. We strictly enforce that agents cannot walk through walls.

We implemented a hierarchical check system:

Walkable Mask: Agents are strictly confined to road pixels derived from shapefiles.

Collision Handling: If a move is invalid, the agent backtracks or slides along the boundary.

JIT Acceleration: We utilized Numba to accelerate this process, allowing us to simulate 10,000 agents over 5,000 steps in just minutes.

[Slide 10: Results]

(动作：切换 PPT，指向数据)

Speaker:
The results of the simulation phase are highly promising.
As shown here, we achieved 100% Road Adherence—meaning no physical violations occurred.
Simultaneously, 96.8% of agents successfully converged to their intended sinks.

This generated dataset of 10,000 high-fidelity trajectories serves as the "Ground Truth" for our next phase.

[Slide 11: Phase 4 - Policy Learning]

(动作：切换 PPT，对比左右两个方框)

Speaker:
You might ask: If the simulation works well, why do we need Phase 4?

The problem is that the Physics Simulation is heavy. It requires storing Gigabytes of potential fields and performing slow gradient queries at every step. It's hard to deploy on edge devices.

Therefore, we use Diffusion Policy Distillation. We treat the simulator as a "Teacher" and train a lightweight Neural Network "Student." This creates a model that is end-to-end, less than 10MB in size, and generalizes well to similar road structures.

[Slide 12: Model Architecture]

(动作：切换 PPT，简述网络结构)

Speaker:
We adopted a 1D-UNet coupled with a Denoising Diffusion Probabilistic Model (DDPM).

We frame trajectory generation as a denoising process. The model takes a 2-frame history (position and velocity) as input and iteratively "denoises" a random sequence into a coherent 8-step future action plan.
This allows the model to capture the multi-modal distribution of human behavior better than simple regression models.

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