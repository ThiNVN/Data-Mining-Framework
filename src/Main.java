import weka.classifiers.evaluation.Evaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.classifiers.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;
import java.util.Random;

import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        //CSV to Arff
        ProcessDataset.csvToArff("src/data/cleaned_weather_data.csv");
        // Load the dataset
        Instances dataset = BuildingClassifier.loadData("src/data/weather_classification.arff");
        //Encode the categorical attributes
        Instances encodedDataset = BuildingClassifier.encodeNominalToBinary(dataset, "5,8,10");
        //Set class attribute for dataset
        dataset.setClassIndex(dataset.numAttributes() - 1);
        //Split trainTest
        Instances[] trainTest = BuildingClassifier.splitTrainTest(dataset);
        Instances train = trainTest[0];
        Instances test = trainTest[1];

        //Step 3: Improve above step2's result by applying clustering and classification
        SimpleKMeans kmeans = BuildingClassifier.clustering(encodedDataset, 3);
        System.out.println(Arrays.toString(kmeans.getClusterSizes()));
        BuildingClassifier.classifyCluster(encodedDataset, kmeans);

        //Build and print J48_Tree classification
        System.out.println("=== J48 ===");
        Classifier tree = BuildingClassifier.J48_tree(train);
        //Evaluate and print J48 classification
        System.out.println("=== J48 evaluation ===");
        Evaluation evalJ48 = BuildingClassifier.evaluateModelMethod(tree, train, test);
        System.out.println("AUC = " + evalJ48.areaUnderPRC(0));
        System.out.println("Precision = " + evalJ48.precision(0));
        System.out.println("Recall = " + evalJ48.recall(0));
        System.out.println("fMeasure = " + evalJ48.fMeasure(0));
        System.out.println("Error rate = " + evalJ48.errorRate() + "\n");
        System.out.println(evalJ48.toMatrixString("=== Overall Confusion Matrix ===\n"));

        //Step 2: Build classification algorithm
        //Build and print OneR classification
        System.out.println("=== OneR ===");
        Classifier oneR = BuildingClassifier.oneR(dataset);
        //Evaluate and print OneR classification
        System.out.println("=== OneR evaluation ===");
        Evaluation evalOneR = BuildingClassifier.evaluateModelMethod(oneR, train, test);
        System.out.println("AUC = " + evalOneR.areaUnderPRC(0));
        System.out.println("Precision = " + evalOneR.precision(0));
        System.out.println("Recall = " + evalOneR.recall(0));
        System.out.println("fMeasure = " + evalOneR.fMeasure(0));
        System.out.println("Error rate = " + evalOneR.errorRate() + "\n");
        System.out.println(evalOneR.toMatrixString("=== Overall Confusion Matrix ===\n"));

        //Build and print NaiveBayes classification
        System.out.println("=== NaiveBayes ===");
        Classifier nb = BuildingClassifier.naiveBayes(dataset);
        System.out.println("=== NaiveBayes evaluation ===");
        Evaluation evalNB = BuildingClassifier.evaluateModelMethod(nb, train, test);
        System.out.println("AUC = " + evalNB.areaUnderPRC(0));
        System.out.println("Precision = " + evalNB.precision(0));
        System.out.println("Recall = " + evalNB.recall(0));
        System.out.println("fMeasure = " + evalNB.fMeasure(0));
        System.out.println("Error rate = " + evalNB.errorRate() + "\n");
        System.out.println(evalOneR.toMatrixString("=== Overall Confusion Matrix ===\n"));

        //Sequential Pattern Mining
        //Remove numeric attributes
        Instances removedDataset = ProcessDataset.removeAttribute(dataset, "1,2,3,4,6,7,9");
        //Apply Apriori to find rules
        BuildingClassifier.aprioriRules(removedDataset);

        //Step 4: Evaluate model using 10-fold cross-validation


    }
}
