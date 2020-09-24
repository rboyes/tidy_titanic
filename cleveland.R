library(tidyverse)
library(tidymodels)
library(randomForest)

df_cleveland <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"), 
                         header = FALSE, 
                         na.strings = c("?")) %>% as_tibble()

colnames(df_cleveland) <- c("age", 
                            "sex", 
                            "cp", 
                            "trestbps", 
                            "chol", 
                            "fbs", 
                            "restecg", 
                            "thalach", 
                            "exang", 
                            "oldpeak", 
                            "slope", 
                            "ca", 
                            "thal", 
                            "target")

df_cleveland <- df_cleveland %>% mutate(target = as.factor(ifelse(target > 0, "heart_disease", "no_heart_disease")))
df_targetdist <- df_cleveland %>% group_by(target) %>% count()

set.seed(5)
split_cleveland <- initial_split(df_cleveland, prop = 0.75, strata = target)
df_training <- split_cleveland %>% training()
df_testing <- split_cleveland %>% testing()

rcp_cleveland <- recipe(target ~ ., data = df_training) %>% 
  step_medianimpute(ca, thal)

prepped_rcp <- rcp_cleveland %>% prep(df_training)

rf_cleveland <- rand_forest(trees = tune(), mtry = tune(), mode = "classification") %>% set_engine("randomForest")

wf_cleveland <- workflow() %>% add_recipe(rcp_cleveland) %>% add_model(rf_cleveland)

rf_params <- wf_cleveland %>% parameters() %>% update(mtry = mtry(range = c(2L, 8L)), trees = trees(range = c(25L, 100L)))

rf_grid <- grid_regular(rf_params, levels = 4)
vfold <- vfold_cv(df_training, v = 4, repeats = 1, strata = target)
rf_search <- tune_grid(wf_cleveland, grid = rf_grid, resamples = vfold, param_info = rf_params)

best_params <- select_best(rf_search, metric = "roc_auc")

wf_final <- finalize_workflow(wf_cleveland, best_params) %>% fit(data = df_training)

df_preds <- wf_final %>% predict(df_testing, type = "prob") %>% bind_cols(df_testing)
df_preds <- wf_final %>% predict(df_testing) %>% bind_cols(df_preds)
df_metrics <- df_preds %>% conf_mat(truth = target, estimate = .pred_class) %>% summary()

df_preds %>% roc_auc(target, .pred_heart_disease)

df_preds %>% roc_curve(target, .pred_heart_disease) %>% autoplot()

rf_final <- pull_workflow_fit(wf_final)
mat_importance <- importance(rf_final$fit)
df_importance <- data.frame(column = rownames(mat_importance), mat_importance)
rownames(df_importance) <- NULL
df_importance %>% arrange(desc(MeanDecreaseGini))

saveRDS(wf_final, "wf_final.rds")