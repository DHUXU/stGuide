# stGuide advances label transfer in spatial transcriptomics through attention-based supervised graph representation learning
## OverView
![图片1](https://github.com/user-attachments/assets/f8aeee7b-ebe9-4116-b9a6-fd9617f9d6e4)
**Overview of stGuide.** a Given multiple ST datasets with three-layer profiles: gene expression in both query and reference, spatial locations, and annotations in the reference data, stGuide annotates query spot labels based on the reference annotations. A composed graph is constructed using -nearest neighbors (KNN) for spatial locations and mutual nearest neighbors (MNN) for transcriptomics within and across slices. b stGuide extracts reference spot representations by aggregating information from spatially and transcriptomic similar spots, guided by spot annotations. c stGuide learns shared representations for the reference and query datasets by aggregating messages from spatially and transcriptomic similar spots, aligning  with the feature space of  to enable label transfer. d The representations of query and reference datasets are used for label transfer and pseudo-time analysis.



