using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using ImageClassification.DataModels;
using static ImageClassification.Model.ConsoleHelpers;
using Common;
using System.Collections;
using System.Collections.Generic;
using static Microsoft.ML.DataOperationsCatalog;

namespace ImageClassification.Model
{
    public class ModelBuilder
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string inputTensorFlowModelFilePath;
        private readonly string outputMlNetModelFilePath;
        private readonly MLContext mlContext;
        private static string LabelAsKey = nameof(LabelAsKey);
        private static string ImageReal = nameof(ImageReal);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        public ModelBuilder(string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.inputTensorFlowModelFilePath = inputModelLocation;
            this.outputMlNetModelFilePath = outputModelLocation;
            mlContext = new MLContext(seed: 1);
        }

        private struct ImageSettingsForTFModel
        {
            public const int imageHeight = 299;        //224 for Inception v1 --- 299 for Inception v3
            public const int imageWidth = 299;         //224 for Inception v1 --- 299 for Inception v3
            public const float mean = 117;             //117 for Inception v1  (offsetImage: ImageSettingsForTFModel.mean)
            public const float scale = 1;              //Not used in ML.NET TF API
            public const bool channelsLast = true;     //true for Inception v1 (interleavePixelColors: ImageSettingsForTFModel.channelsLast)
        }

        public void BuildAndTrain(IEnumerable<ImageData> imageSet, IEnumerable<ImageData> testImageSet)
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {inputTensorFlowModelFilePath}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageSettingsForTFModel.imageWidth},{ImageSettingsForTFModel.imageHeight}), image mean: {ImageSettingsForTFModel.mean}");

            // 1. Load images information (filenames and labels) in IDataView

            //Single Full Image Set
            //
            //IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(imageSet);
            //IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            //// Split the data 75:25 into train and test sets, train and evaluate.
            //TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.25);
            //IDataView trainDataView = trainTestData.TrainSet;
            //IDataView testDataView = trainTestData.TestSet;

            //Load seggregated train-image-set 
            IDataView trainDataView = mlContext.Data.LoadFromEnumerable(imageSet);
            //Load seggregated test-image-set 
            IDataView testDataView = mlContext.Data.LoadFromEnumerable(testImageSet);

            // 2. Load images in-memory while applying image transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelAsKey, inputColumnName: "Label")
                            .Append(mlContext.Transforms.LoadImages(outputColumnName: "image_object", imageFolder: imagesFolder, inputColumnName: nameof(DataModels.ImageData.ImagePath)))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image_object_resized", imageWidth: ImageSettingsForTFModel.imageWidth, imageHeight: ImageSettingsForTFModel.imageHeight, inputColumnName: "image_object"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", inputColumnName: "image_object_resized", interleavePixelColors: ImageSettingsForTFModel.channelsLast, offsetImage: ImageSettingsForTFModel.mean))
                            .Append(mlContext.Model.LoadTensorFlowModel(inputTensorFlowModelFilePath).
                                 ScoreTensorFlowModel(outputColumnNames: new[] { "InceptionV3/Predictions/Reshape" }, 
                                                      inputColumnNames: new[] { "input" }, 
                                                      addBatchDimensionInput: false));  // (For Inception v1 --> addBatchDimensionInput: true)  (For Inception v3 --> addBatchDimensionInput: false)
            // Input and output column names have to coincide with the input and output tensor names of the TensorFlow model
            // You can check out those tensor names by opening the Tensorflow .pb model with a visual tool like Netron: https://github.com/lutzroeder/netron
            // TF .pb model --> input node --> INPUTS --> input --> id: "input" 
            // TF .pb model --> Softmax node --> INPUTS --> logits --> id: "softmax2_pre_activation" (Inceptionv1) or "InceptionV3/Predictions/Reshape" (Inception v3)

            // 

            // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations 
            ConsoleHelper.PeekDataViewInConsole(mlContext, trainDataView, dataProcessPipeline, 2);
            //ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "InceptionV3/Predictions/Reshape", trainDataView, dataProcessPipeline, 2);

            // 3. Set the training algorithm and convert back the key to the categorical values                            
            var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelAsKey, featureColumnName: "InceptionV3/Predictions/Reshape");  //"softmax2_pre_activation" for Inception v1
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"));

            // 4. Train the model
            // Measuring training time
            var watch = System.Diagnostics.Stopwatch.StartNew();

            ConsoleWriteHeader("Training the ML.NET classification model");
            ITransformer model = trainingPipeline.Fit(trainDataView);

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine("Training with transfer learning took: " + (elapsedMs / 1000).ToString() + " seconds");

            // 5. Make bulk predictions and calculate quality metrics
            ConsoleWriteHeader("Create Predictions and Evaluate the model quality");
            IDataView predictionsDataView = model.Transform(testDataView);
           
            // This is an optional step, but it's useful for debugging issues
            var loadedModelOutputColumnNames = predictionsDataView.Schema
                .Where(col => !col.IsHidden).Select(col => col.Name);

            // 5.1 Show the predictions
            ConsoleWriteHeader("*** Showing all the predictions ***");
            List<ImageWithPipelineFeatures> predictions = mlContext.Data.CreateEnumerable<ImageWithPipelineFeatures>(predictionsDataView, false, true).ToList();
            predictions.ForEach(pred => ConsoleWriteImagePrediction(pred.ImagePath, pred.Label, pred.PredictedLabelValue, pred.Score.Max()));

            // 5.2 Show the performance metrics for the multi-class classification            
            var classificationContext = mlContext.MulticlassClassification;
            ConsoleWriteHeader("Classification metrics");
            var metrics = classificationContext.Evaluate(predictionsDataView, labelColumnName: LabelAsKey, predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            // 6. Save the model to assets/outputs
            ConsoleWriteHeader("Save model to local file");
            ModelHelpers.DeleteAssets(outputMlNetModelFilePath);

            mlContext.Model.Save(model, predictionsDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved: {outputMlNetModelFilePath}");
        }

    }
}
