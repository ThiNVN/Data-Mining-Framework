import weka.associations.Apriori;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.meta.*;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.Filter;

import java.util.Random;

public class BuildingClassifier {

    //Load the dataset source
    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances dataset = source.getDataSet();
//        data.setClassIndex(data.numAttributes() - 1);
        return dataset;
    }

    //Naive bayes
    public static Classifier naiveBayes(Instances dataset) throws Exception {
        dataset.setClassIndex(dataset.numAttributes() - 1);
        //create and build classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);
        //print out capabilities
        System.out.println(nb.getCapabilities().toString());
        return nb;
    }

    //J48 Decision tree
    public static Classifier J48_tree(Instances dataset) throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(dataset);
        System.out.println(tree.getCapabilities().toString());
        System.out.println(tree.graph());
        return tree;
    }

    //OneR
    public static Classifier oneR(Instances dataset) throws Exception {
        OneR oneR = new OneR();
        oneR.buildClassifier(dataset);
        System.out.println(oneR.toString());
        return oneR;
    }

    //ZeroR
    public static Classifier zeroR(Instances dataset) throws Exception {
        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(dataset);
        System.out.println(zeroR.toString());
        return zeroR;
    }

    //Random Tree
    public static Classifier randomTree(Instances dataset) throws Exception {
        RandomTree tree = new RandomTree();
        tree.buildClassifier(dataset);
        System.out.println(tree.getCapabilities().toString());
        return tree;
    }

    //Random Forest
    public static Classifier randomForest(Instances dataset) throws Exception {
        RandomForest rf = new RandomForest();
        rf.buildClassifier(dataset);
        System.out.println(rf.toString());
        return rf;
    }

    //Filter the training set
    public void classifierWithFilter(Instances dataset, weka.classifiers.Classifier classifier) throws Exception {
        Remove remove = new Remove();
        String[] opts = new String[]{"-R", "1"};
        remove.setOptions(opts);

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(remove);
        fc.setClassifier(classifier);
        fc.buildClassifier(dataset);

        System.out.println(classifier.getCapabilities().toString());

        if (classifier instanceof J48) {
            System.out.println(((J48) classifier).graph());
        }
    }

    //Linear Regression model (optional)
    public void linearRegression(Instances dataset) throws Exception {
        //Set class attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);
        //Build linear regression model
        LinearRegression lr = new LinearRegression();
        lr.buildClassifier(dataset);
        //Print out linear regression model
        System.out.println(lr);
    }

    //Sequential minimal optimization (optional)
    public void SMO(Instances dataset) throws Exception {
        //Set class attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);
        //Build SMO regression model
        SMOreg smo = new SMOreg();
        smo.buildClassifier(dataset);
        //Print out SMO regression model
        System.out.println(smo);
    }

    //Evaluate Model
    public static Evaluation evaluateModelMethod(weka.classifiers.Classifier classifier, weka.core.Instances training_set, weka.core.Instances test_set) throws Exception {
        Evaluation eval = new Evaluation(training_set);
        //Run evaluation
        eval.evaluateModel(classifier, test_set);

        //Print evaluation
        System.out.println(eval.toSummaryString("Evaluation result:\n", true));
        /*
        System.out.println("AUC = " + eval.areaUnderPRC(0));
        System.out.println("Precision = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error rate = " + eval.errorRate() + "\n");
        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
        */
        return eval;
    }

    //Cross-validation using 10 folds
    public static void evaluateModelFolds(weka.classifiers.Classifier classifier, Instances training_set, Instances test_set) throws Exception {
        training_set.setClassIndex(training_set.numAttributes() - 1);
        //Initialize
        int seed = 1;
        int folds = 10;
        Random rand = new Random(seed);

        //Test set
        test_set.setClassIndex(test_set.numAttributes() - 1);
        //Create random dataset
        Instances randData = new Instances(training_set);
        randData.randomize(rand);
        //Stratify
        if (randData.classAttribute().isNominal()) {
            randData.stratify(folds);
        }

        //Perform cross-validation
        for (int i = 0; i < folds; i++) {
            Evaluation eval = new Evaluation(randData);
            //Get the folds
            Instances train = randData.trainCV(folds, i);
            Instances test = randData.testCV(folds, i);
            //Evaluate classifier
            eval.evaluateModel(classifier, test);

            //Print evaluation
            System.out.println(eval.toSummaryString("Evaluation result:\n", false));
            /*
            System.out.println("AUC = " + eval.areaUnderPRC(0));
            System.out.println("Precision = " + eval.precision(0));
            System.out.println("Recall = " + eval.recall(0));
            System.out.println("fMeasure = " + eval.fMeasure(0));
            System.out.println("Error rate = " + eval.errorRate() + "\n");
            System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
            */
        }

    }

    //Make prediction
    public void predictInstance(weka.classifiers.Classifier classifier, Instances training_set, Instances test_set) throws Exception {
        training_set.setClassIndex(training_set.numAttributes() - 1);
        int num_classes = training_set.numClasses();
        //Print out class values
        for (int i = 0; i < num_classes; i++) {
            String classValue = training_set.classAttribute().value(i);
            System.out.println("Class value " + i + ": " + classValue);
        }
        classifier.buildClassifier(training_set);

        //Test set
        test_set.setClassIndex(test_set.numAttributes() - 1);
        //Loop through the new dataset and make predictions
        System.out.println("=================");
        System.out.println("Actual Class, Classifier Predicted");
        for (int i = 0; i < test_set.numAttributes(); i++) {
            //Get class double value for current instance
            double actualClass = test_set.instance(i).classValue();
            //Get class string value using the class index using the class's int value
            String actualClassValue = test_set.classAttribute().value((int) actualClass);
            //Get instance object of current instance
            Instance newInstance = test_set.instance(i);
            //Call classifyInstance
            double predInstance = classifier.classifyInstance(newInstance);
            //Use this value to get string value of the predicted class
            String predString = test_set.classAttribute().value((int) predInstance);
            System.out.println(actualClass + ": " + predString);
        }
    }

    //SaveModelFile
    public void saveModel(weka.classifiers.Classifier classifier, String filename) throws Exception {
        weka.core.SerializationHelper.write(filename, classifier);
    }
    //Load Model
    public void loadModel(weka.classifiers.Classifier classifier, String filename) throws Exception {
        classifier = (weka.classifiers.Classifier) weka.core.SerializationHelper.read(filename);
    }

    //Aggregation
    public void aggregation(Instances training_set) throws Exception {
        training_set.setClassIndex(training_set.numAttributes() - 1);

        //AdaBoost
        AdaBoostM1 m1 = new AdaBoostM1();
        m1.setClassifier(new DecisionStump());
        m1.setNumIterations(100);
        m1.buildClassifier(training_set);

        //Bagging
        Bagging bagging = new Bagging();
        bagging.setClassifier(new RandomTree());
        bagging.setNumIterations(25);
        bagging.buildClassifier(training_set);

        //Stacking
        Stacking stacking = new Stacking();
        stacking.setMetaClassifier(new Logistic());
        weka.classifiers.Classifier[] classifiers = {new J48(), new NaiveBayes(), new RandomForest()};
        stacking.setClassifiers(classifiers);
        stacking.buildClassifier(training_set);

        //Voting
        Vote vote = new Vote();
        vote.setClassifiers(classifiers);
        vote.buildClassifier(training_set);
    }

    //Association rules
    public static void aprioriRules(Instances dataset) throws Exception {
        //Apriori model
        Apriori apriori = new Apriori();
        //Build model
        apriori.buildAssociations(dataset);
        //Print out extracted rules
        System.out.println(apriori);
    }

    //Clustering
    public static weka.clusterers.SimpleKMeans clustering(Instances dataset, int numberOfClusters) throws Exception {
        //New instance of clusterer
        SimpleKMeans kmeans = new SimpleKMeans();
        //Number of clusters
        kmeans.setNumClusters(numberOfClusters);
        //Set distance function
        kmeans.setDistanceFunction(new weka.core.EuclideanDistance());
        //Build clusterer
        kmeans.buildClusterer(dataset);
        System.out.println(kmeans);

        //Evaluate clusterer
        ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
        clusterEvaluation.setClusterer(kmeans);
        clusterEvaluation.evaluateClusterer(dataset);
        System.out.println(clusterEvaluation.clusterResultsToString());

        return kmeans;
    }

    //Classify clusters
    public static void classifyCluster(Instances dataset, weka.clusterers.SimpleKMeans kmeans) throws Exception {
        dataset.setClassIndex(dataset.numAttributes() - 1);
        for (Instance instance : dataset) {
            int clusterIndex = kmeans.clusterInstance(instance);
        }
        for (int i = 0; i < kmeans.getNumClusters(); i++) {
            Instances clusterInstances = new Instances(dataset, 0);
            for (Instance instance : dataset) {
                if(kmeans.clusterInstance(instance) == i) {
                    clusterInstances.add(instance);
                }
            }
            Classifier nb = J48_tree(clusterInstances);

            // Shuffle the dataset for randomness
            clusterInstances.randomize(new Random(1)); // Seed for reproducibility

            // Split percentage for training (70% training, 30% testing)
            int trainSize = (int) Math.round(clusterInstances.numInstances() * 0.7);
            int testSize = clusterInstances.numInstances() - trainSize;

            // Generate training and testing sets
            Instances train = new Instances(clusterInstances, 0, trainSize);
            Instances test = new Instances(clusterInstances, trainSize, testSize);

            evaluateModelMethod(nb, train, test);
        }
    }

    //Clustering with ignored attribute
    public static weka.clusterers.SimpleKMeans clusteringWithIgnored(Instances dataset, int numberOfClusters, String activeAttribute) throws Exception {
        //New instance of clusterer
        SimpleKMeans kmeans = new SimpleKMeans();
        //Number of clusters
        kmeans.setNumClusters(numberOfClusters);
        //Set distance function
        EuclideanDistance distanceFunction = new EuclideanDistance();
        distanceFunction.setAttributeIndices(activeAttribute);
        kmeans.setDistanceFunction(distanceFunction);
        //Build clusterer
        kmeans.buildClusterer(dataset);
        System.out.println(kmeans);

        //Evaluate clusterer
        ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
        clusterEvaluation.setClusterer(kmeans);
        clusterEvaluation.evaluateClusterer(dataset);
        System.out.println(clusterEvaluation.clusterResultsToString());

        return kmeans;
    }

    //Encode nominal
    public static Instances encodeNominalToBinary(Instances dataset, String encodeAttribute) throws Exception {
        //Create Filter
        NominalToBinary nominalToBinary = new NominalToBinary();
        //Set index of attribute to be encoded
        nominalToBinary.setAttributeIndices(encodeAttribute);
        //Input the dataset into the Filter
        nominalToBinary.setInputFormat(dataset);
        //Apply Filter to dataset
        Instances encodedInstances = Filter.useFilter(dataset, nominalToBinary);
        return encodedInstances;
    }

}
