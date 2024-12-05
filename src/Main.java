import weka.classifiers.evaluation.Evaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.classifiers.*;
import ca.pfv.spmf.algorithms.clustering.hierarchical_clustering.AlgoHierarchicalClustering;

import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        //CSV to Arff
        ProcessDataset.csvToArff("src/data/cleaned_weather_data.csv");
        // Load the dataset
        Instances dataset = BuildingClassifier.loadData("src/data/weather_classification.arff");
        //Encode the nominal attributes
        Instances encodedDataset = BuildingClassifier.encodeNominalToBinary(dataset, "5,8,10");

        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Shuffle the dataset for randomness
        dataset.randomize(new Random(1)); // Seed for reproducibility

        // Split percentage for training (70% training, 30% testing)
        int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
        int testSize = dataset.numInstances() - trainSize;

        // Generate training and testing sets
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);

        //Step 3: Improve above step2's result by applying clustering and classification
        SimpleKMeans kmeans = BuildingClassifier.clustering(encodedDataset, 3);
        System.out.println(Arrays.toString(kmeans.getClusterSizes()));
        BuildingClassifier.classifyCluster(encodedDataset, kmeans);

        //Step 2: Build classification algorithm
        //Build and print classifier
        Classifier tree = BuildingClassifier.J48_tree(train);
        //Evaluate and print classifier's evaluation
        Evaluation eval = BuildingClassifier.evaluateModelMethod(tree, train, test);
        System.out.println("AUC = " + eval.areaUnderPRC(0));
        System.out.println("Precision = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error rate = " + eval.errorRate() + "\n");
        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));

        //Step 4: Evaluate model using 10-fold cross-validation
        System.out.println("=== 10-fold cross-validation ===");
        BuildingClassifier.evaluateModelFolds(tree, train, test);

    }
}
