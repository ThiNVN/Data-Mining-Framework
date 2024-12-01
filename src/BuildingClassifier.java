import weka.associations.Apriori;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.meta.*;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class BuildingClassifier {

    //Load the dataset source
    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    //Naive bayes
    public static Classifier naiveBayes(Instances dataset) throws Exception {
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
    public static void evaluateModelMethod(weka.classifiers.Classifier classifier, weka.core.Instances training_set, weka.core.Instances test_set) throws Exception {
        Evaluation eval = new Evaluation(training_set);
        //Run evaluation
        eval.evaluateModel(classifier, test_set);

        //Print evaluation
        System.out.println(eval.toSummaryString("Evaluation result:\n", true));
        /*
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Incorrect % = " + eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderPRC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error rate = " + eval.errorRate() + "\n");
        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
        */

    }

    //Cross-validation using 10 folds
    public void evaluateModelFolds(weka.classifiers.Classifier classifier, Instances training_set, Instances test_set) throws Exception {
        training_set.setClassIndex(training_set.numAttributes() - 1);
        //Initilize
        int seed = 1;
        int folds = 10;
        Random rand = new Random(1);

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
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Incorrect % = " + eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderPRC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
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
    public void associationRules(Instances dataset) throws Exception {
        //Apriori model
        Apriori apriori = new Apriori();
        //Build model
        apriori.buildAssociations(dataset);
        //Print out extracted rules
        System.out.println(apriori);
    }
}
