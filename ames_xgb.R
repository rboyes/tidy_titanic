library(tidymodels)
library(tune)
library(AmesHousing)
library(xgboost)
library(yardstick)

library(doParallel)

# ------------------------------------------------------------------------------

ames <- make_ames()

ames <- ames %>% mutate(Sale_Price = log(Sale_Price, base = 10))

set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price", prop = 0.8)

df_train <- training(data_split)
df_test <- testing(data_split)

set.seed(2453)
rs_splits <- vfold_cv(df_train, strata = "Sale_Price", v = 4)

# ------------------------------------------------------------------------------

ames_rec <-
  recipe(Sale_Price ~ ., data = ames_train) %>%
  step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = .1)  %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_ns(Longitude, deg_free = 2) %>%
  step_ns(Latitude, deg_free = 2)

xgb_spec_simple <- boost_tree(trees = 250) %>% set_engine("xgboost") %>% set_mode("regression")
wf_ames_simple <- workflow() %>% add_recipe(ames_rec) %>% add_model(xgb_spec_simple)
wf_final <- wf_ames_simple %>% fit(df_train)
ames_preds <- wf_final %>% predict(df_test) %>% bind_cols(df_test)
yardstick::rmse(ames_preds, .pred, Sale_Price)
yardstick::rmse(ames_preds %>% mutate(prediction = 10^.pred, Sale_Price = 10^Sale_Price) %>% select(prediction, Sale_Price), prediction, Sale_Price)



xgb_spec <- boost_tree(
  trees = 250, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

xgb_grid <- grid_random(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), ames_train),
  learn_rate(),
  size = 100
)

wf_ames <- workflow() %>% add_recipe(ames_rec) %>% add_model(xgb_spec)

wf_ames_simple <- workflow() %>% add_recipe(ames_rec) %>% add_model(xgb_spec_simple)

wf_final <- wf_ames_simple %>% fit(df_train)

gs_ames <-
  tune_grid(
    wf_ames,
    resamples = rs_splits,
    grid = xgb_grid,
    control = control_grid(verbose = TRUE)
  )

wf_final <- finalize_workflow(wf_ames, select_best(gs_ames, metric = 'rmse')) %>% 
  fit(data = ames_train)

ames_preds <- wf_final %>% predict(df_test) %>% bind_cols(df_test)

df_preds <- wf_final %>% predict(df_test, type = "prob") %>% bind_cols(df_test)
df_preds <- wf_final %>% predict(df_test) %>% bind_cols(df_preds)

