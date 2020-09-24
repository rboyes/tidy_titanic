library(tidyverse)
library(tidymodels)
library(titanic)
library(xgboost)


tbl_train <- tibble(titanic_train) %>% mutate(Survived = as.factor(Survived))
tbl_test <- tibble(titanic_test)


rcp_transform <- recipe(Survived ~ ., tbl_train) %>%
  step_mutate(Title = str_extract(Name, "Mr\\.|Mrs\\.|Miss\\.|Master\\.")) %>%
  step_mutate(Title = if_else(is.na(Title), "Unknown", Title)) %>%
  step_mutate(Pclass = factor(Pclass),
              Sex = factor(Sex),
              SibSp = factor(SibSp),
              Embarked = factor(Embarked),
              Title = factor(Title)) %>%
  step_dummy(Pclass, Sex, SibSp, Embarked, Title) %>%
  step_medianimpute(Age, Fare) %>%
  step_rm(PassengerId, Name, Ticket, Cabin)

prep(rcp_transform, tbl_train) %>% bake(tbl_test)

xgb_spec_simple <- boost_tree(trees = 100) %>% set_engine("xgboost") %>% set_mode("classification")

xgb_spec <- boost_tree(
  trees = 500, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), tbl_train),
  learn_rate(),
  size = 30
)

wf_titanic <- workflow() %>% add_recipe(rcp_transform) %>% add_model(xgb_spec)

vfold <- vfold_cv(tbl_train, v = 4, repeats = 1, strata = Survived)

xgb_res <- tune_grid(wf_titanic, 
                     resamples = vfold, 
                     grid = xgb_grid, 
                     control = control_grid(save_pred = TRUE, 
                                            verbose = TRUE))

best_params <- select_best(xgb_res, metric = "roc_auc")

wf_final <- finalize_workflow(wf_titanic, best_params) %>% fit(data = tbl_train)

tbl_preds <- wf_final %>% predict(tbl_test, type = "prob") %>% bind_cols(tbl_test)
tbl_preds <- wf_final %>% predict(tbl_test) %>% bind_cols(tbl_preds)

xgb_final <- pull_workflow_fit(wf_final)
tbl_featimp <- xgb.importance(model = xgb_final$fit)

