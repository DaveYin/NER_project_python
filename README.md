# NER_project_python
This is a project about NER (named entity recognition) for Chinese social media.

The deep model we have built is an attention-lstm model.

A linear-chain CRF is used to tag the feature vectors.

The system gets a rawdata file, extracts the entities in each segment, and exports the results to a result file.

There are five types of entities which can be recognized by the extractor.(Person, Loction, Organization, Company, Product)

The corpus is from https://github.com/mswellhao/chineseNER/tree/master/KG/workDir/data.
