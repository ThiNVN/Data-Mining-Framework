package labdatamining;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class WekaEvaluation {
    public static void main(String[] args) {
        try {
            String datasetPath = "C:\\Users\\Admin\\weather_classification_data.arff";
            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            J48 tree = new J48();
            tree.setOptions(new String[]{"-C", "0.25", "-M", "2"});

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 10, new Random(1));

            System.out.println("=== Evaluation Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println("=== Classifier Accuracy ===");
            System.out.println("Correctly Classified Instances: " + eval.pctCorrect() + "%");
            System.out.println("Incorrectly Classified Instances: " + eval.pctIncorrect() + "%");
            System.out.println("=== Detailed Class-wise Evaluation ===");
            System.out.println(eval.toClassDetailsString());
            System.out.println("=== Confusion Matrix ===");
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
