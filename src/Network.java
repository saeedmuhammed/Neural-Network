import java.util.ArrayList;
import java.io.*;
import java.util.Random;


public class Network {

    public static void main(String[] args) throws IOException {
        int inputNodes=0;
        int outputNodes=0;
        int hiddenNodes=0;
        int sizeOfData=0;
        double[][] X;
        double[][] Y;
        double[][] Weightshidden;//                 {{0.3,-0.9,1.0},{-1.2,1.0,1.0}};;
        double[][] Weightsoutput;//              {{1.0,0.8}};
        Layer hiddenLayer = new Layer();
        Layer outputLayer = new Layer();
        Layer inputLayer = new Layer();
        double meanError=0.0;









        FileInputStream fstream = new FileInputStream("train.txt");
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String strLine;
        int lines=0;
        strLine = br.readLine();
        String[] tokens = strLine.trim().split(" +");
        inputNodes=Integer.parseInt(tokens[0]);
        hiddenNodes=Integer.parseInt(tokens[1]);
        outputNodes=Integer.parseInt(tokens[2]);
        strLine = br.readLine();
        tokens = strLine.trim().split(" +");
        sizeOfData=Integer.parseInt(tokens[0]);
        //inputNodes++; hiddenNodes++; for bias
        X = new double[sizeOfData][inputNodes];
        Y = new double[sizeOfData][outputNodes];
        Weightshidden=new double[hiddenNodes][inputNodes];
        Weightsoutput=new double[outputNodes][hiddenNodes];

        /*add Bias*/

       /* for (int i = 0; i <sizeOfData ; i++) {
            X[i][0]=1;
        }*/




        while ((strLine = br.readLine()) != null)   {

             tokens = strLine.trim().split(" +");
            /* sperate x and Y*/
                    for (int j = 0; j <tokens.length ; j++) {
                        if(j>=tokens.length-outputNodes){ // to save output in y if it one or more output
                            Y[lines][j-(tokens.length-outputNodes)]=Double.parseDouble(tokens[j]);

                        }
                        else {
                            X[lines][j]=Double.parseDouble(tokens[j]); //make x[lines][j+1] for bias
                        }
                    }
                lines++;

            }


        in.close();




        double[][] newX =normalizeData(X,sizeOfData,inputNodes);
        //get min and max in output
        double maxNum = Y[0][0];
        double minNum = Y[0][0];
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[i].length; j++) {
                if(maxNum < Y[i][j]){
                    maxNum = Y[i][j];
                }else if(minNum > Y[i][j]){
                    minNum = Y[i][j];
                }
            }
        }
        //normalize output between 0 and 1
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[i].length; j++) {
                Y[i][j]=(Y[i][j]-minNum) / (maxNum-minNum);

            }
        }



        Random rd = new Random();
        for (int i = 0; i <hiddenNodes ; i++) {
            for (int j = 0; j <inputNodes ; j++) {

                Weightshidden[i][j]=-2+ rd.nextDouble() * (2-(-2));
            }

        }

        for (int i = 0; i <outputNodes ; i++) {
            for (int j = 0; j <hiddenNodes ; j++) {
                Weightsoutput[i][j]=-2+ rd.nextDouble() * (2-(-2));
            }

        }
        

        

        for (int i = 0; i <500 ; i++) {
              double error=0.0;
            for (int j = 0; j <sizeOfData ; j++) {
                /*Feeedforwad start*/
                for (int k = 0; k <hiddenNodes ; k++) {
                    double sum=0.0;
                    for (int l = 0; l <inputNodes ; l++) {
                        sum+=Weightshidden[k][l] * newX[j][l];


                    }

                    Neuron neuron = new Neuron();
                    neuron.value=sigmoid(sum);
                    hiddenLayer.Neurons.add(neuron);


                }

                for (int k = 0; k <outputNodes ; k++) {
                    double sum=0.0;
                    for (int l = 0; l <hiddenNodes ; l++) {
                        sum+=Weightsoutput[k][l] * hiddenLayer.Neurons.get(l).value;

                    }
                    Neuron neuron = new Neuron();
                    neuron.value=sigmoid(sum);
                    outputLayer.Neurons.add(neuron);
                    //System.out.println(Y[j][k]);
                    //System.out.println(sigmoid(sum));
                    error+= Math.pow(Y[j][k] - sigmoid(sum),2);

                }
                /*Feeedforwad end*/

                for (int k = 0; k <outputNodes ; k++) {
                    outputLayer.Neurons.get(k).propagatioError=(outputLayer.Neurons.get(k).value-Y[j][k]) * outputLayer.Neurons.get(k).value *(1-outputLayer.Neurons.get(k).value);

                }

                for (int k = 0; k <hiddenNodes ; k++) {
                      double sum=0.0;
                    for (int l = 0; l <outputNodes ; l++) {
                        sum+=outputLayer.Neurons.get(l).propagatioError * Weightsoutput[l][k];
                    }
                    hiddenLayer.Neurons.get(k).propagatioError=sum * hiddenLayer.Neurons.get(k).value * (1-hiddenLayer.Neurons.get(k).value);

                }

                for (int k = 0; k <hiddenNodes ; k++) {
                    for (int l = 0; l <outputNodes ; l++) {
                        Weightsoutput[l][k] = Weightsoutput[l][k] - 0.001 * outputLayer.Neurons.get(l).propagatioError * hiddenLayer.Neurons.get(k).value;

                    }

                }
                for (int k = 0; k <inputNodes ; k++) {
                    for (int l = 0; l <hiddenNodes ; l++) {
                        Weightshidden[l][k] = Weightshidden[l][k] - 0.001 * hiddenLayer.Neurons.get(l).propagatioError * newX[j][k];
                       // System.out.println(Weightshidden[l][k]);
                    } 
                    
                }


            }

             meanError = error/sizeOfData;

            if(meanError<=0.5){break;}


            }
        System.out.println(meanError);

        FileWriter myWriter = new FileWriter("weights.txt");

        myWriter.write("HiddenLayer Weights \n");
        for (int i = 0; i <hiddenNodes ; i++) {
            for (int j = 0; j <inputNodes ; j++) {
                myWriter.write(String.valueOf(Weightshidden[i][j]) + " ");
            }
            myWriter.write("\n\n");


        }
        myWriter.write("\nOutputLayer Weights \n");
        for (int i = 0; i <outputNodes ; i++) {
            for (int j = 0; j <hiddenNodes ; j++) {
                myWriter.write(String.valueOf(Weightsoutput[i][j]) + " ");
            }
            myWriter.write("\n\n");


        }



        myWriter.close();

      ArrayList<Double> testOutput = predict(Weightshidden,Weightsoutput);


        }

        



public static ArrayList<Double> predict(double[][] Weightshidden , double[][] Weightsoutput) throws IOException {
    int inputNodes=0;
    int outputNodes=0;
    int hiddenNodes=0;
    int sizeOfData=0;
    double[][] X;
    double[][] Y;
    double meanError=0.0;

    FileInputStream fstream = new FileInputStream("test.txt");
    DataInputStream in = new DataInputStream(fstream);
    BufferedReader br = new BufferedReader(new InputStreamReader(in));
    String strLine;
    int lines=0;
    strLine = br.readLine();
    String[] tokens = strLine.trim().split(" +");
    inputNodes=Integer.parseInt(tokens[0]);
    hiddenNodes=Integer.parseInt(tokens[1]);
    outputNodes=Integer.parseInt(tokens[2]);
    strLine = br.readLine();
    tokens = strLine.trim().split(" +");
    sizeOfData=Integer.parseInt(tokens[0]);
    X = new double[sizeOfData][inputNodes];
    Y = new double[sizeOfData][outputNodes];
    Weightshidden=new double[hiddenNodes][inputNodes];
    Weightsoutput=new double[outputNodes][hiddenNodes];
    ArrayList<Double> hiddenLayerValues = new ArrayList<>();
    ArrayList<Double> outputLayerValues = new ArrayList<>();



    while ((strLine = br.readLine()) != null)   {

        tokens = strLine.trim().split(" +");
        /* sperate x and Y*/
        for (int j = 0; j <tokens.length ; j++) {
            if(j>=tokens.length-outputNodes){
                Y[lines][j-(tokens.length-outputNodes)]=Double.parseDouble(tokens[j]);

            }
            else {
                X[lines][j]=Double.parseDouble(tokens[j]);
            }
        }
        lines++;

    }


    in.close();

    double maxNum = Y[0][0];
    double minNum = Y[0][0];
    for (int i = 0; i < Y.length; i++) {
        for (int j = 0; j < Y[i].length; j++) {
            if(maxNum < Y[i][j]){
                maxNum = Y[i][j];
            }else if(minNum > Y[i][j]){
                minNum = Y[i][j];
            }
        }
    }

    //normalize output between 0 and 1
    for (int i = 0; i < Y.length; i++) {
        for (int j = 0; j < Y[i].length; j++) {
            Y[i][j]=(Y[i][j]-minNum) / (maxNum-minNum);
           // System.out.println(Y[i][j]);

        }
    }

    double error=0.0;
    for (int j = 0; j <sizeOfData ; j++) {
        /*Feeedforwad start*/
        for (int k = 0; k <hiddenNodes ; k++) {
            double sum=0.0;
            for (int l = 0; l <inputNodes ; l++) {
                sum+=Weightshidden[k][l] * X[j][l];


            }

            hiddenLayerValues.add(sigmoid(sum));



        }

        for (int k = 0; k <outputNodes ; k++) {
            double sum=0.0;
            for (int l = 0; l <hiddenNodes ; l++) {
                sum+=Weightsoutput[k][l] * hiddenLayerValues.get(l);

            }
            outputLayerValues.add(sigmoid(sum));
            error+= Math.pow(Y[j][k] - sigmoid(sum),2);

        }



}
    meanError = error/sizeOfData;
    System.out.println(meanError);
   return outputLayerValues;
    }

public static double sigmoid(double z){
    return 1 / (1 + Math.exp(-z));
}

 public static double[][] normalizeData(double[][] x, int sizeOfData, int inputNodes){
           double[][] newX = new double[sizeOfData][inputNodes] ;
         for (int i = 0; i < inputNodes; i++) {
            double[] col = getCols(x,i,sizeOfData);
            double mean = calculateMean(col,sizeOfData);
            double standardDevisation = calculateStandardDeviation(mean,col,sizeOfData);
             for (int j = 0; j <sizeOfData ; j++) {
                 col[j]=(col[j]-mean)/standardDevisation;
                 double s=col[j];
                 newX[j][i]=s;
             }

         }
return newX;



}

public static double[] getCols(double[][] x, int colNum, int sizeOfdata){
     double[] col = new double[sizeOfdata];
    for (int i = 0; i <sizeOfdata ; i++) {
        col[i]=x[i][colNum];

    }

return col;


}
public static double calculateMean(double[] col, int sizeOfData) {

    double sum = 0;

    for (int i = 0; i < sizeOfData; i++) {

            sum+=col[i];

    }
    return sum / sizeOfData;
}

public static double calculateStandardDeviation(double mean, double[] col, int sizeOfData){

    double sum=0;
    for (int i = 0; i <sizeOfData ; i++) {

            sum += Math.pow((col[i] - mean), 2);
    }

return Math.sqrt(sum);
}




}
