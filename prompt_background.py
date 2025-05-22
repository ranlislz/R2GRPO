# Prompt backgrounds
# ner_background = """Extract specific entities from the following sentence. The entities to be identified are: 'Dataset', 'Task', and 'Method'.

# ### Entity Definitions:
# - 'Task': A task in machine learning refers to the specific problem or type of problem that a ML/AI model/method is designed to solve. Tasks can be broad, like classification, regression, or clustering, or they can be very specific, such as Pedestrian Detection, Autonomous Driving, Sentiment Analysis, Named Entity Recognition, and Relation Extraction.
# - 'Method': A method entity refers to the approach, algorithm, or technique used to solve a specific task/problem. Methods encompass the computational algorithms, model architectures, and the training procedures that are employed to make predictions or decisions based on data. For example, Convolutional Neural Networks, Dropout, data augmentation, recurrent neural networks.
# - 'Dataset': A realistic collection of data that is used for training, validating, or testing the algorithms. These datasets can consist of various forms of data such as text, images, videos, or structured data. For example, MNIST, COCO, AGNews, IMDb.

# ### Other Notes:
# - Generics cannot be used independently to refer to any specific entities, e.g., 'This task', 'the dataset', and 'a public corpus' are not entities.
# - The determiners should not be part of an entity span. For example, given span 'the SQuAD v1.1 dataset', where the determiner 'the' should be excluded from the entity span.
# - If both the full name and the abbreviation are present in the sentence, annotate the abbreviation and its corresponding full name separately. For instance, '20-newsgroup (20NG)'.
# - Only annotate "factual, content-bearing" entities. Task, dataset, and method entities normally have specific names and their meanings are consistent across different papers. For example, "CoNLL03", "SNLI" are factual entities. Annotators should annotate only the minimum necessary to represent the original meaning of task/dataset/metric (e.g., "The", "dataset", "public", 'method', 'technique' are often omitted).
# - Treat multi-word phrases as single entities if they form a specific name or concept (e.g., "Convolutional Neural Networks" is one "Method", not "Convolutional" and "Neural Networks").
# - Do not label standalone generic terms like "dataset", "method", "task", "model", or "features" as entities unless part of a specific name (e.g., "IMDb dataset" is a "Dataset", but "dataset" alone is not).
# - Ensure entity types match their definitions: "Task" for ML problems (e.g., "Sentiment Analysis"), "Method" for techniques (e.g., "BERT"), "Dataset" for data collections (e.g., "MNIST").
# - Avoid overlapping entities; select the longest valid span (e.g., "deep learning model" should be one "Method", not "deep learning" and "model" separately).
# - Do not extract entities that are pronouns (e.g., "it", "they") or vague references; focus on explicit names or terms.
# """

ner_background = """Extract specific entities from the following sentence. The entities to be identified are: 'Dataset', 'Task', and 'Method'.

### Entity Definitions:
- 'Task': A task in machine learning refers to the specific problem or type of problem that a ML/AI model/method is designed to solve. Tasks can be broad, like classification, regression, or clustering, or they can be very specific, such as Pedestrian Detection, Autonomous Driving, Sentiment Analysis, Named Entity Recognition, and Relation Extraction.
- 'Method': A method entity refers to the approach, algorithm, or technique used to solve a specific task/problem. Methods encompass the computational algorithms, model architectures, and the training procedures that are employed to make predictions or decisions based on data. For example, Convolutional Neural Networks, Dropout, data augmentation, recurrent neural networks.
- 'Dataset': A realistic collection of data that is used for training, validating, or testing the algorithms. These datasets can consist of various forms of data such as text, images, videos, or structured data. For example, MNIST, COCO, AGNews, IMDb.

### Other Notes:
- Generics cannot be used independently to refer to any specific entities, e.g., 'This task', 'the dataset', and 'a public corpus' are not entities.
- The determiners should not be part of an entity span. For example, given span 'the SQuAD v1.1 dataset', where the determiner 'the' should be excluded from the entity span.
- If both the full name and the abbreviation are present in the sentence, annotate the abbreviation and its corresponding full name separately. For instance, '20-newsgroup (20NG)'.
- Only annotate "factual, content-bearing" entities. Task, dataset, and method entities normally have specific names and their meanings are consistent across different papers. For example, "CoNLL03", "SNLI" are factual entities. Annotators should annotate only the minimum necessary to represent the original meaning of task/dataset/metric (e.g., "The", "dataset", "public", 'method', 'technique' are often omitted).
"""

re_background = """Based on the given sentence and the entities with their types, determine the relationship between each pair. The potential relations are: ['Part-Of', 'SubClass-Of', 'SubTask-Of', 'Benchmark-For', 'Trained-With', 'Evaluated-With', 'Synonym-Of', 'Used-For', 'Compare-With']. If no relationship exists between a pair, do not include it in the output.

### Relationship Definitions:
- 'Part-Of': This relationship denotes that one entity (e.g., a Method) is a component or a part of another entity (e.g., another Method).
- 'SubClass-Of': Specifies that one entity is a subclass or a specialized version of another entity.
- 'SubTask-Of': Indicates that one Task is a subset or a specific aspect of another broader Task.
- 'Benchmark-For': Shows that a Dataset serves as a standard or benchmark for evaluating the performance of a Method on a Task.
- 'Trained-With': Indicates that a Method is trained using a Dataset.
- 'Evaluated-With': This relationship denotes that a Method is evaluated using a Dataset to test its performance or conduct experiments.
- 'Synonym-Of': Indicates that two entities are considered to have the same or very similar meaning, such as abbreviations.
- 'Used-For': Shows that one entity (e.g., a Method) is utilized for achieving or performing another entity (e.g., a Task). This relationship is highly flexible.
- 'Compare-With': This relationship is used when one entity is compared with another to highlight differences, similarities, or both.

### Notes:
- Determine the 'Relationship' that best describes how the subject and object are related, based on the sentence context.
- Please do not annotate negative relations (e.g., X is not used in Y).
- Annotate a relationship only if there is direct evidence or clear implication in the text. Avoid inferring relationships that are not explicitly mentioned or clearly implied.
"""

re_golden_background = """Based on the given sentence and the provided subject and object entity pairs with their types, determine the relationship between each pair. The potential relations are: ['Part-Of', 'SubClass-Of', 'SubTask-Of', 'Benchmark-For', 'Trained-With', 'Evaluated-With', 'Synonym-Of', 'Used-For', 'Compare-With']. If no relationship exists between a pair, do not include it in the output.

### Entity Types:
- 'Task': A task in machine learning refers to the specific problem or type of problem that a ML/AI model/method is designed to solve (e.g., object detection).
- 'Method': A method entity refers to the approach, algorithm, or technique used to solve a specific task/problem (e.g., CornerNet).
- 'Dataset': A realistic collection of data used for training, validating, or testing algorithms (e.g., MS COCO).

### Relationship Definitions:
- 'Part-Of': This relationship denotes that one entity (e.g., a Method) is a component or a part of another entity (e.g., another Method).
- 'SubClass-Of': Specifies that one entity is a subclass or a specialized version of another entity.
- 'SubTask-Of': Indicates that one Task is a subset or a specific aspect of another broader Task.
- 'Benchmark-For': Shows that a Dataset serves as a standard or benchmark for evaluating the performance of a Method on a Task.
- 'Trained-With': Indicates that a Method is trained using a Dataset.
- 'Evaluated-With': This relationship denotes that a Method is evaluated using a Dataset to test its performance or conduct experiments.
- 'Synonym-Of': Indicates that two entities are considered to have the same or very similar meaning, such as abbreviations.
- 'Used-For': Shows that one entity (e.g., a Method) is utilized for achieving or performing another entity (e.g., a Task). This relationship is highly flexible.
- 'Compare-With': This relationship is used when one entity is compared with another to highlight differences, similarities, or both.

### Notes:
- Determine the 'Relationship' that best describes how the subject and object are related, based on the sentence context.
- Please do not annotate negative relations (e.g., X is not used in Y).
- Annotate a relationship only if there is direct evidence or clear implication in the text. Avoid inferring relationships that are not explicitly mentioned or clearly implied.
"""

re_plus_background = """Based on the given sentence and the provided list of extracted entities with their types, extract relationship triplets. Each triplet consists of one subject entity, one object entity, and their relationship. The interested entity types are: ['Dataset', 'Method', 'Task']. The potential relations are: ['Part-Of', 'SubClass-Of', 'SubTask-Of', 'Benchmark-For', 'Trained-With', 'Evaluated-With', 'Synonym-Of', 'Used-For', 'Compare-With'].

### Entity Definitions:
- 'Task': A task in machine learning refers to the specific problem or type of problem that a ML/AI model/method is designed to solve. Tasks can be broad, like classification, regression, or clustering, or they can be very specific, such as Pedestrian Detection, Autonomous Driving, Sentiment Analysis, Named Entity Recognition, and Relation Extraction.
- 'Method': A method entity refers to the approach, algorithm, or technique used to solve a specific task/problem. Methods encompass the computational algorithms, model architectures, and the training procedures that are employed to make predictions or decisions based on data. For example, Convolutional Neural Networks, Dropout, data augmentation, recurrent neural networks.
- 'Dataset': A realistic collection of data that is used for training, validating, or testing the algorithms. These datasets can consist of various forms of data such as text, images, videos, or structured data. For example, MNIST, COCO, AGNews, IMDb.

### Relationship Definitions:
- 'Part-Of': This relationship denotes that one method is a component or a part of another method.
- 'SubClass-Of': Specifies that one method is a subclass or a specialized version of another method.
- 'SubTask-Of': Indicates that one task is a subset or a specific aspect of another broader task.
- 'Benchmark-For': Shows that a dataset serves as a standard or benchmark for evaluating the performance of methods on a specific task.
- 'Trained-With': Indicates that a method is trained using a specific dataset.
- 'Evaluated-With': This relationship denotes that a method is evaluated using a specific dataset to test its performance or conduct experiments.
- 'Synonym-Of': Indicates that two terms or entities are considered to have the same or very similar meaning, such as abbreviations.
- 'Used-For': Shows that one entity is utilized for achieving or performing another entity. For example, one Method is Used-For one Task. This relationship is highly flexible, allowing for generic relationships across diverse entities.
- 'Compare-With': This relationship is used when one entity is compared with another to highlight differences, similarities, or both.

### Notes:
- Only annotate "factual, content-bearing" entities and relationships.
- If there are no relationship triplets, return "NULL".
"""

ner_background_erc = """Extract specific entities from the following sentence. The entities to be identified are: 'Task', 'Method', 'Metric', 'Material', 'Other-ScientificTerm', and 'Generic'.

### Entity Definitions:
- 'Task': Applications, problems to solve, or systems to construct in a scientific context. Examples include information extraction, machine reading system, image segmentation.
- 'Method': Methods, models, systems to use, tools, components of a system, or frameworks. Examples include language model, CORENLP, POS parser, kernel method.
- 'Metric': Metrics, measures, or entities that express the quality of a system or method. Examples include F1, BLEU, Precision, Recall, ROC curve, mean reciprocal rank, mean-squared error, robustness, time complexity.
- 'Material': Data, datasets, resources, corpora, or knowledge bases. Examples include image data, speech data, stereo images, bilingual dictionary, paraphrased questions, CoNLL, Penn Treebank, WordNet, Wikipedia.
- 'Other-ScientificTerm': Phrases that are scientific terms but do not fall into the above categories. Examples include physical or geometric constraints, qualitative prior knowledge, discourse structure, syntactic rule, tree, node, tree kernel, features, noise, criteria.
- 'Generic': General terms or pronouns that may refer to an entity but are not themselves informative, often used as connection words. Examples include model, approach, prior knowledge, them, it. Note: Only annotate 'Generic' entities if they are involved in a relation.

### Notes:
- Do not include determiners (e.g., 'the', 'a') or adjective pronouns (e.g., 'this', 'its') in the entity span.
- Only annotate factual, content-bearing entities with consistent meanings (e.g., 'CoNLL', 'language model', not 'the system'), except for 'Generic' entities which are tagged only when part of a relation.
- If both full name and abbreviation are present, annotate them separately (e.g., 'Penn Treebank' and 'PTB').
- Follow the ACL RD-TEC Annotation Guideline for entity boundary annotation, allowing embedded spans if the shorter span is involved in a relation."""

re_background_erc = """### Relationship Definitions:
- 'Compare': A symmetric relation where the subject and object are compared. Examples: 'Unlike the quantitative prior, the qualitative prior is often ignored', 'We compare our system with previous sequential tagging systems'.
- 'Part-of': The subject is a component or part of the object (directed: subject → object). Examples: 'The system includes two models', 'We incorporate NLU module to the system'.
- 'Conjunction': A symmetric relation where the subject and object function similarly or are used together. Examples: 'obtained from human expert or knowledge base', 'NLP applications such as machine translation and language generation'.
- 'Evaluate-for': The subject (typically a Metric) evaluates the object (e.g., a Method or Task) (directed: subject → object). Example: 'F1 evaluates the performance of the language model'.
- 'Feature-of': The subject is a feature, property, or belongs to the object (directed: subject → object). Examples: 'prior knowledge of the model', 'genre-specific regularities of discourse structure'.
- 'Used-for': The subject is used for the object (directed: subject → object). Examples: 'The TISPER system has been designed to enable many text applications', 'Our method models user proficiency', 'Our algorithms exploit local smoothness'.
- 'Hyponym-Of': The subject is a specific type or instance of the object (directed: subject → object). Examples: 'TUIT is a software library', 'NLP applications such as machine translation and language generation'.

### Notes:
- Relations do not cross sentence boundaries.
- Do not annotate negative relations (e.g., 'X is not used in Y').
- Annotate a relationship only if there is direct evidence or clear implication in the sentence.
- For asymmetric relations (Part-of, Evaluate-for, Feature-of, Used-for, Hyponym-Of), the relation is directed from subject to object. Compare and Conjunction are symmetric."""