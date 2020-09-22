library(tidyverse)
library(tidymodels)
library(titanic)



df_train <- titanic_train
df_test <- titanic_test
df_test <- df_test %>% filter(PassengerId != 1306)

df_train <- df_train %>% mutate(Survived = as.factor(Survived))

rcp_transform <- recipe(Survived ~ ., df_train) %>%
  step_mutate(Title = str_extract(Name, "Mr\\.|Mrs\\.|Miss\\.|Master\\.")) %>%
  step_mutate(Title = if_else(is.na(Title), "Unknown", Title)) %>%
  step_mutate(Pclass = factor(Pclass),
              Sex = factor(Sex),
              SibSp = factor(SibSp),
              Embarked = factor(Embarked),
              Title = factor(Title)) %>%
  step_medianimpute(Age, Fare) %>%
  step_rm(PassengerId, Name, Ticket, Cabin)

prep(rcp_transform, df_train) %>% bake(df_test)

rf_titanic <- rand_forest(trees = tune(), mtry = tune(), mode = "classification") %>% set_engine("randomForest")

wf_titanic <- workflow() %>% add_recipe(rcp_transform) %>% add_model(rf_titanic)

rf_params <- wf_titanic %>% parameters() %>% update(mtry = mtry(range = c(2L, 8L)), trees = trees(range = c(25L, 100L)))

rf_grid <- grid_regular(rf_params, levels = 4)
vfold <- vfold_cv(df_train, v = 4, repeats = 1, strata = Survived)
rf_search <- tune_grid(wf_titanic, grid = rf_grid, resamples = vfold, param_info = rf_params)
best_params <- select_best(rf_search, metric = "roc_auc")

wf_final <- finalize_workflow(wf_titanic, best_params) %>% fit(data = df_train)

df_preds <- wf_final %>% predict(df_test, type = "prob") %>% bind_cols(df_test)
df_preds <- wf_final %>% predict(df_test) %>% bind_cols(df_preds)

rf_final <- pull_workflow_fit(wf_final)
mat_importance <- importance(rf_final$fit)
df_importance <- data.frame(column = rownames(mat_importance), mat_importance)
rownames(df_importance) <- NULL
df_importance %>% arrange(desc(MeanDecreaseGini))
