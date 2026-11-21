# Supplementary computational-profile data

`Supplementary_Table_S1_Learning_Runtime_8Networks.xlsx` contains the measured learning-based computational profile for the eight main real-world networks.

The workbook has two worksheets:

- `Summary`: network-level means, standard deviations, graph sizes, preprocessing, epidemic-threshold values, training infection probabilities, device information, and repeat counts;
- `Raw`: all measured and derived timing records.

The reported phases are:

- **1D-CGS inference:** frozen-model prediction only, after one untimed warm-up, with five measured runs;
- **SGNN training:** network-specific SIR/feature preparation plus model initialization and optimization, using three independent model seeds;
- **SGNN neural inference:** one untimed warm-up plus five measured runs for each trained model;
- **SGNN structural-entropy reranking:** one untimed warm-up plus five measured runs for each trained model;
- **SGNN total:** training plus the mean neural-inference and structural-entropy reranking times for the corresponding model seed.

The workbook contains measured runtime data only. It does not contain or redistribute the external 1D-CGS or SGNN implementations. Their original publications and DOI links are listed in the repository README.
