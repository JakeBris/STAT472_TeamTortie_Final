library(ranger)
library(pROC)
library(dplyr)
library(tidyr)
library(ggplot2)
library(knitr)
library(gt)
library(gtExtras)
library(catboost)
library(caret)
library(vip)

adultdf <- read.csv('adultdf.csv')

# Convert integer variables to factors as necessary
adultdf <- adultdf %>%
  mutate(across(-c(agep, weight, height, pov_ratio, sleep_hours), as.factor))


# Splitting train and test data
set.seed(472)
n <- nrow(adultdf)
indices <- sample(1:n, replace = FALSE)

k <- floor(0.8 * n)

train_data <- adultdf[indices[1:k], ]
test_data  <- adultdf[indices[(k+1):n], ]

# Omitting columns with substantial missing values
train_data <- train_data %>%
  select(-bmi_cat, -cover_65, -cover_u65, -lastdr_vis, -retail_vis, -numretail_vis, -urg_vis, -numurg_vis, -emerg_vis, -numemer_vis, -ovrhosp_vis,
         -covtype_chip, -covtype_none, -covtype_oth, -covtype_ss, -covtype_ihs, -covtype_mil, -covtype_maid, -covtype_mgap, -covtype_mcare, -covtype_priv,
         -prost_can, - uter_can, - ovary_can, -cerv_can, -ssi_inc, -dib_type, -hdnck_can, -thyro_can, -throat_can, -stom_can, -skndk_can, -sknnm_can, -sknm_can, 
         -rect_can, -pan_can, -mouth_can, -melan_can, -lymph_can, -lung_can, -liver_can, -leuk_can, -laryn_can, -gall_can, -esoph_can, -breast_can, -brain_can, 
         -bone_can, -blood_can, -other_can, -colrcc_can, -colon_can, -bladd_can, -nowther_mh, -wic_inc, -gesdib_ever, -walk_leis, -skiprx_cost, -delayrx_cost, -lessrx_cost)

test_data <- test_data %>%
  select(-bmi_cat, -cover_65, -cover_u65, -lastdr_vis, -retail_vis, -numretail_vis, -urg_vis, -numurg_vis, -emerg_vis, -numemer_vis, -ovrhosp_vis,
         -covtype_chip, -covtype_none, -covtype_oth, -covtype_ss, -covtype_ihs, -covtype_mil, -covtype_maid, -covtype_mgap, -covtype_mcare, -covtype_priv,
         -prost_can, - uter_can, - ovary_can, -cerv_can, -ssi_inc, -dib_type, -hdnck_can, -thyro_can, -throat_can, -stom_can, -skndk_can, -sknnm_can, -sknm_can, 
         -rect_can, -pan_can, -mouth_can, -melan_can, -lymph_can, -lung_can, -liver_can, -leuk_can, -laryn_can, -gall_can, -esoph_can, -breast_can, -brain_can, 
         -bone_can, -blood_can, -other_can, -colrcc_can, -colon_can, -bladd_can, -nowther_mh, -wic_inc, -gesdib_ever, -walk_leis, -skiprx_cost, -delayrx_cost, -lessrx_cost)

# Random Forest: All Predictors
rf <- ranger(
  anymedcare ~ .,
  data = train_data, 
  num.trees = 500,
  mtry = 9,
  importance = 'impurity',
  classification = T,
  probability = T
)

predicts <- predict(rf, data = test_data)
rf_preds <- predicts$predictions
pred_probs <- rf_preds[, 2]
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

confusionMatrix(as.factor(pred_class), as.factor(test_data$anymedcare))
rf_accuracy <- 0.6172 # Balanced accuracy output from confusionMatrix above

roc_curve <- roc(test_data$anymedcare, pred_probs)
auc(roc_curve)
roc_df <- data.frame(specificity = roc_curve$specificities, sensitivity = roc_curve$sensitivities)
ggroc(roc_curve, legacy.axes = T) + 
  theme_minimal() + 
  labs(title = "ROC Curve for Logistic Model, all predictors", color = "blue") + 
  geom_abline(slope = 1, intercept = 0,  linetype = "dashed", color = "gray50", linewidth = 0.8) + 
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_curve), 3)),
           size = 3, fontface = "bold", color = "#2C7BB6")

vip(rf, num_features = 10) + geom_col(fill = 'steelblue', width = 0.7) + labs(title = "Feature Importance", subtitle = "Top 10 Predictors of Seeking Medical Care") + theme_minimal()

# Logistic Regression: All Predictors
logit_model <- glm(
  anymedcare ~ .,
  data = train_data, 
  family = binomial(link = "logit")
)

summary(logit_model)

logit_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_class <- ifelse(logit_probs > 0.5, "1", "0")

confusionMatrix(as.factor(logit_class), as.factor(test_data$anymedcare))
logit_accuracy <- 0.6139 # Balanced Accuracy from confusionMatrix


roc_logit <- roc(test_data$anymedcare, logit_probs)

ggroc(roc_logit, legacy.axes = T) + theme_minimal() + labs(title = "ROC Curve: LR Model, all predictors", color = "blue") + geom_abline(slope = 1, intercept = 0, 
                                                                                                                linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_logit), 3)),
           size = 3, fontface = "bold", color = "#2C7BB6")

# Combined ROC curves for LR and RF
rocs_org <- list("Logistic Regression" = roc_logit,  "Random Forest" = roc_curve)
ggroc(rocs_org, legacy.axes = T) + theme_minimal() + labs(title = "ROC Curves for LR and RF", color = "Model") + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("L.R. AUC = ", round(auc(roc_logit), 3), "\n R.F. AUC= ", round(auc(roc_curve), 3), ""),
           size = 1.4, fontface = "bold", color = "#2C7BB6")
                                                                                                                                 
### CatBoost

# CatBoost Model 1: All 93 features
cat_features <- train_data %>%
  select(c(-anymedcare, -weight, -height, -agep, -pov_ratio, -sleep_hours))

train_pool <- catboost.load_pool(data = train_data[, -72], label = train_data[,72])
test_pool <- catboost.load_pool(data = test_data[, -72], label = test_data[, 72])

params <- list(
  loss_function = "Logloss", 
  iterations = 1000,
  depth = 10, 
  verbose = 100,
  l2_leaf_reg = 6
)
cat_model <- catboost.train(train_pool, params = params)

pred_probs <- catboost.predict(cat_model, test_pool, prediction_type = "Probability")
pred_labels <- catboost.predict(cat_model, test_pool, prediction_type = "Class")

actual <- test_data$anymedcare
predicted <- pred_labels
accuracy <- sum(actual == predicted) / length(actual)
accuracy 

feature_importance <- catboost.get_feature_importance(cat_model, pool = train_pool)

preds_train <- catboost.predict(cat_model, train_pool, prediction_type = "Probability")
preds_test <- catboost.predict(cat_model, test_pool, prediction_type = "Probability")

roc_test <- roc(test_data[,72], preds_test)

ggroc(roc_test, legacy.axes = T) + theme_minimal() + labs(title = "ROC Curve for CATBoost, all features", color = "blue") + geom_abline(slope = 1, intercept = 0, 
                                                                                                                                      linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_test), 3)),
           size = 3, fontface = "bold", color = "#2C7BB6")


# CatBoost Model 2: Omitting Mental Health Features
train_data2 <- train_data %>%
  select(-c(life_satis, effort_mh, anx_mh, dep_mh, sad_mh, tired_mh, 
            lonely_mh, dlyther_mh, hopeless_mh, nervous_mh, needther_mh, pastther_mh, restless_mh, 
            pyschdist_mh, worthless_mh, use_rxmh))
test_data2 <- test_data %>%
  select(-c(life_satis, effort_mh, anx_mh, dep_mh, sad_mh, tired_mh, 
            lonely_mh, dlyther_mh, hopeless_mh, nervous_mh, needther_mh, pastther_mh, restless_mh, 
            pyschdist_mh, worthless_mh, use_rxmh))

cat_features2 <- train_data2 %>%
  select(c(-anymedcare, -weight, -height, -agep, -pov_ratio, -sleep_hours))

train_pool2 <- catboost.load_pool(data = train_data2[, -72], label = train_data2[,72])
test_pool2 <- catboost.load_pool(data = test_data2[, -72], label = test_data2[, 72])

params <- list(
  loss_function = "Logloss", 
  iterations = 1000,
  depth = 6, 
  verbose = 100,
  l2_leaf_reg = 6
)
cat_model2 <- catboost.train(train_pool2, params = params)

pred_probs2 <- catboost.predict(cat_model2, test_pool2, prediction_type = "Probability")
pred_labels2 <- catboost.predict(cat_model2, test_pool2, prediction_type = "Class")

actual2 <- test_data2$anymedcare
predicted2 <- pred_labels2
accuracy2 <- sum(actual == predicted2) / length(actual2)
accuracy2 

feature_importance2 <- catboost.get_feature_importance(cat_model2, pool = train_pool2)

preds_train2 <- catboost.predict(cat_model2, train_pool2, prediction_type = "Probability")
preds_test2 <- catboost.predict(cat_model2, test_pool2, prediction_type = "Probability")

roc_test2 <- roc(test_data[,72], preds_test2)

ggroc(roc_test2, legacy.axes = T) + theme_minimal() + labs(title = "ROC Curve for CATBoost, omit MH", color = "blue") + geom_abline(slope = 1, intercept = 0, 
                                                                                                                                     linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_test2), 3)),
           size = 3, fontface = "bold", color = "#2C7BB6")

# CatBoost Model 3: Top 10 most important features
train_data3 <- train_data %>%
  select(pov_ratio, agep, weight, height, educ_status, drink_status, sleep_hours, use_rx, region, urb_rural, anymedcare)
test_data3 <- test_data %>%
  select(pov_ratio, agep, weight, height, educ_status, drink_status, sleep_hours, use_rx, region, urb_rural, anymedcare)

train_pool3 <- catboost.load_pool(data = train_data3[, -11], label = train_data3[,11])
test_pool3 <- catboost.load_pool(data = test_data3[, -11], label = test_data3[, 11])


params <- list(
  loss_function= "Logloss", 
  iterations = 1000,
  depth = 6, 
  verbose = 100,
  l2_leaf_reg = 6
)
cat_model3 <- catboost.train(train_pool3, params = params)

pred_probs3 <- catboost.predict(cat_model3, test_pool3, prediction_type = "Probability")
pred_labels3 <- catboost.predict(cat_model3, test_pool3, prediction_type = "Class")

actual3 <- test_data3$anymedcare
predicted3 <- pred_labels3
accuracy3 <- sum(actual == predicted3) / length(actual3)
accuracy3 

preds_train3 <- catboost.predict(cat_model3, train_pool3, prediction_type = "Probability")
preds_test3 <- catboost.predict(cat_model3, test_pool3, prediction_type = "Probability")


feature_importance3 <- catboost.get_feature_importance(cat_model3, pool = train_pool3)
roc_test3 <- roc(test_data3[,11], preds_test3)


ggroc(roc_test3, legacy.axes = T) + theme_minimal() + labs(title = "ROC Curve for CATBoost, top 10 features", color = "blue") + geom_abline(slope = 1, intercept = 0, 
                                                                                                                                    linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_test3), 3)),
           size = 3, fontface = "bold", color = "#2C7BB6")


# CATBoost models' ROC curves
rocs <- list("All 93 Predictors" = roc_test, "Omit Mental Health" = roc_test2, "Top 10 Features" = roc_test3)
ggroc(rocs, legacy.axes = T) + theme_minimal() + labs(title = "Multiple ROC Curves for CatBoost", color = "Model") + geom_abline(slope = 1, intercept = 0, 
                                                                                                                linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste0("AUC = ", round(auc(roc_test), 3)),
           size = 1.75, fontface = "bold", color = "#F8766D") +
  annotate("text", x = 0.75, y = 0.18, 
           label = paste0("AUC = ", round(auc(roc_test2), 3)),
           size = 1.75, fontface = "bold", color = "#00BA38") +
  annotate("text", x = 0.75, y = 0.12, 
           label = paste0("AUC = ", round(auc(roc_test3), 3)),
           size = 1.75, fontface = "bold", color = "#619CFF")

# Table Summary of Model Accuracies
summary_table_models <- data.frame(Method = c('Logistic Regression, all predictors', 'Random Forest, all predictors', 'CAT Boost, all predictors', 'CAT Boost, omit mental health predictors', 
                                              'CAT Boost, only top 10 predictors'),
                                   Accuracy = c(logit_accuracy, rf_accuracy, accuracy, accuracy2, accuracy3))

summary_table_models %>% gt() %>% gt_theme_pff() %>% fmt_number(columns = Accuracy, decimals = 3)

# Table of CATBoost Parameters
cat_params <- data.frame(Parameter = c('Loss Function', 'Iterations', 
                                       'Depth', 'Verbose', 'L2 regularization'), Description = c('What the model is trying to minimze', 'Number of boosting steps', 'Maximum depth of each tree', 'How often CatBoost prints training progess', 'Penalizes large leaf values, thereby preventing overfitting'), Values = c('Logloss', 1000, 6, 100, 6))
cat_params %>% gt() %>% gt_theme_pff()
