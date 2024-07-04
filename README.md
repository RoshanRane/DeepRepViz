# DeepRepViz: Identifying Potential Confounders in Deep Learning Model Predictions
## Overview
[$\textit{DeepRepViz}$](https://deep-rep-viz.vercel.app/) is an interactive web tool that can be used to inspect the 3D representations learned by predictive deep learning (DL) models. It can be used to inspect a trained model for any biases in the training and the data and debug the model. The tool is intended to help improve the transparency of such DL models. The DL models can be a 'black box' and hinder their adaptation to critical decision processes (ex: medical diagnosis). This tool aims to provide a platform that developers of DL models can use to gain a better intuition about what their model is learning by visualizing its 'learned representation' of the data. It can be used to understand what the model might be basing its decisions upon. <br>

## How does $\textit{DeepRepViz}$ work?
$\textit{DeepRepViz}$ coprises two components - an online visualization tool and a metric called the 'Con-score'. To learn more about DeepRepViz, you could read '[DeepRepViz: Identifying Confounders in Deep Learning Model Predictions](https://arxiv.org/abs/2309.15551)'

![Overview](https://github.com/ritterlab/DeepRepViz/assets/39021807/405ae15c-d94e-48a1-a5fd-f3bd7a23a1ea)

## How to get strated?
Read these:

- [DeepRepViz document](https://deep-rep-viz.vercel.app/docs.html)
- [DeepRpeViz tutorial]() (TBA)

## Acknowlegdements and Funding

This project was inspired by Google Brain's [projector.tensorflow.org](https://projector.tensorflow.org/), but is more catering towards the medical domain and medical imaging analysis. For implementation, we heavily rely on [3D-scatter-plot from plotly.js](https://plotly.com/javascript/3d-scatter-plots/).

This project was funded by the DeSBi Research Unit (DFG; KI-FOR 5363; Profject ID 459422098), the consortium SFB/TRR 265 Losing and Regaining Control over Drug Intake (DFG; Project ID 402170461), FONDA (DFG; SFB 1404; Project ID: 414984028) and FOR 5187 (DFG; Project ID: 442075332).

## Citation

```bash
@misc{rane2024deeprepvizidentifyingconfoundersdeep,
  title={DeepRepViz: Identifying Confounders in Deep Learning Model Predictions}, 
  author={Roshan Prakash Rane and JiHoon Kim and Arjun Umesha and Didem Stark and Marc-André Schulz and Kerstin Ritter},
  year={2024},
  eprint={2309.15551},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2309.15551},
}
```