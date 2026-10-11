# Dataset setup

The notebook expects an authorized copy of the prepared dataset at `data/twitter_emotion.csv`, containing:

| Column | Description |
| --- | --- |
| `text` | Original text sample |
| `label` | Label among anger, fear, joy, love, sadness, surprise |
| `label_id` | Original integer label |
| `split` | train, validation, or test |

The prepared data used in this research contain 416,809 labeled examples with 333,447 train, 41,681 validation, and 41,681 test samples.

**The original raw text dataset is not redistributed in this public repository.** To execute the existing notebook, obtain or prepare the data through an authorized source and place it at the relative path above. The notebook's recorded outputs can be read without rerunning the code.
