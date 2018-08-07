#Exemplo retirado de: https://topepo.github.io/caret/model-training-and-tuning.html

# Stochastich gradient boost: 
# Overview do  eh uma tecnica de machine learning que pode ser usada para classificacao e regressao, produz um modelo do tipo 
#ensemble de classificadores fracos, tipicamente arvores de decisao, otimiza uma funcao de perda diferenciavel (como outros classificadores)
# Foi baseado no Gradiente boost com a diferenca que em cada etapa de treinamento, um classificador base seja treinado com uma amostra 
#do conjunto de teste aleatoriamente sem reposicao

# Pacotes usados
library(mlbench)
library(caret)

data(Sonar)
str(Sonar[, 1:10])

#cria particoes de treino/testes aleatorias
set.seed(998)
inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training <- Sonar[ inTraining,]
testing  <- Sonar[-inTraining,]

#como sera feito o treinamento
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

#Treina com selecao de folds aleatorias
set.seed(825)
gbmFit3 <- train(Class ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid,
                 ## Specify which metric to optimize
                 metric = "ROC")

#Isso eh um modelo!!!!!
print(gbmFit3)

#Aqui usamos o modelo pronto
print(predict(gbmFit3, newdata = head(testing), type = "prob"))
