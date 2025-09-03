import json
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class IntentRecognizer:
    """
    A simple intent recognition model that uses cosine similarity to classify user questions.
    It provides a focused prompt for the agent based on the recognized intent to prevent "over-prompting".
    """
    def __init__(self, examples_file: str = "config/intents.json", prompts_file: str = "config/prompts.yaml"):
        """
        Initializes the recognizer by loading examples and training a TF-IDF vectorizer.

        Args:
            examples_file (str): Path to the JSON file containing example questions.
            prompts_file (str): Path to the YAML file containing task-specific prompts.
        """
        self.intent_examples = self._load_examples(examples_file)
        self.vectorizer = TfidfVectorizer().fit(self._get_all_examples())
        self.example_vectors = self.vectorizer.transform(self._get_all_examples())
        self.intent_labels = self._get_all_labels()
        
        # Load task-specific prompts from YAML
        with open(prompts_file, "r") as f:
            prompts = yaml.safe_load(f)
        self.task_specific_prompts = prompts["task_specific_prompts"]

    def _load_examples(self, examples_file: str) -> dict:
        """Loads the JSON file with intent examples."""
        with open(examples_file, "r") as f:
            return json.load(f)

    def _get_all_examples(self) -> list:
        """Returns a flat list of all example questions."""
        examples = []
        for intent in self.intent_examples:
            examples.extend(self.intent_examples[intent])
        return examples

    def _get_all_labels(self) -> list:
        """Returns a flat list of all intent labels corresponding to the examples."""
        labels = []
        for intent in self.intent_examples:
            labels.extend([intent] * len(self.intent_examples[intent]))
        return labels

    def recognize_intent(self, question: str) -> str:
        """
        Recognizes the intent of a new question by finding the most similar example.

        Args:
            question (str): The user's new question.

        Returns:
            str: The name of the recognized intent.
        """
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.example_vectors)[0]
        
        # Find the index of the most similar example
        best_match_index = np.argmax(similarities)
        
        # Get the intent label of the best match
        recognized_intent = self.intent_labels[best_match_index]
        
        # You could also add a confidence threshold here
        # if similarities[best_match_index] < 0.5:
        #     return "general_query"
        
        return recognized_intent

    def get_task_specific_prompt(self, intent: str) -> str:
        """
        Returns a focused prompt for a given intent.

        Args:
            intent (str): The recognized intent name.

        Returns:
            str: The task-specific prompt.
        """
        return self.task_specific_prompts.get(intent, "")

class PlotRecognizer:
    """
    A recognizer to identify if a user question is asking for a plot.
    """
    def __init__(self):
        self.plot_intent_examples = {
            "needs_plot": [
                "show me a plot", "show pca", "plot pc3 vs pc4", "create a graph", 
                "visualize the data", "plot the expression", "make a chart", "generate a heatmap", 
                "show correlation plot", "create volcano plot", "plot the distribution", 
                "show me histogram", "compare visually", "display as chart",
                "what does the expression pattern look like", "show differences graphically",
                "plot gene expression across samples", "create scatter plot", "show bar chart",
                "visualize the results", "generate visualization", "display correlation matrix",
                "show pathway enrichment plot", "create PCA plot", "plot MDS coordinates"
            ],
            "no_plot": [
                "what is the expression level", "how many genes are upregulated", "list the top genes",
                "what is the p-value", "show me the metadata", "which samples are available",
                "what tables do you have", "hello", "hi there", "thanks", "explain the results",
                "what does this mean", "describe the analysis", "what is differential expression",
                "how was this calculated", "what are the statistics", "give me the numbers",
                "list all samples", "count the genes", "what is the fold change for gene X",
                "hi", "how are you", "what can you do", "tell me a joke", "what is your name",
                "what is the weather today", "how old are you", "what is the meaning of life"
            ]
        }

        # Train vectorizer for plot intent classification
        plot_examples = self._get_all_plot_examples()
        self.plot_vectorizer = TfidfVectorizer().fit(plot_examples)
        self.plot_example_vectors = self.plot_vectorizer.transform(plot_examples)
        self.plot_intent_labels = self._get_all_plot_labels()

    def _get_all_plot_examples(self) -> list:
        """Returns a flat list of all plot intent example questions."""
        examples = []
        for intent in self.plot_intent_examples:
            examples.extend(self.plot_intent_examples[intent])
        return examples

    def _get_all_plot_labels(self) -> list:
        """Returns a flat list of all plot intent labels corresponding to the examples."""
        labels = []
        for intent in self.plot_intent_examples:
            labels.extend([intent] * len(self.plot_intent_examples[intent]))
        return labels

    def needs_visualization(self, question: str) -> bool:
        """
        Determines if a question needs visualization using TF-IDF similarity.
        
        Args:
            question (str): The user's question.
            
        Returns:
            bool: True if visualization is needed, False otherwise.
        """
        question_vector = self.plot_vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.plot_example_vectors)[0]
        
        # Find the index of the most similar example
        best_match_index = np.argmax(similarities)
        
        # Get the intent label of the best match
        plot_intent = self.plot_intent_labels[best_match_index]
        
        # Optional: Add confidence threshold
        confidence = similarities[best_match_index]
        if confidence < 0.1:  # Very low similarity, default to no plot
            return False
        
        return plot_intent == "needs_plot"


if __name__ == '__main__':
    # Example usage:
    recognizer = IntentRecognizer(examples_file="config/intents.json", prompts_file="config/prompts.yaml")
    plotter_recognizer = PlotRecognizer()

    # User questions to test
    test_questions = [
        "what is the expression level of MYC in sample s3?",
        "what are the top upregulated genes in my experiment?",
        "can you show me the correlation of my samples?",
        "hi",
        "what kind of data tables do you have available?",
        "plot the expression of GAPDH across all samples",
        "show pca by patient"
    ]

    for q in test_questions:
        print(f"User question: '{q}'")
        intent = recognizer.recognize_intent(q)
        prompt = recognizer.get_task_specific_prompt(intent)
        plot = plotter_recognizer.needs_visualization(q)
        print(f"Recognized intent: '{intent}'")
        print(f"--- Task-Specific Prompt ---\n{prompt}\n---------------------------\n")
        print(f"Needs visualization: {plot}\n")
        print("====================================\n")
