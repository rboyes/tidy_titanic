library(tidymodels)
library(tune)
library(AmesHousing)

# ------------------------------------------------------------------------------

ames <- make_ames()

ames <- ames %>% mutate(Sale_Price = log(Sale_Price, base = 10))

set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price", prop = 1/2)

ames_train <- training(data_split)
ames_test <- testing(data_split)

set.seed(2453)
rs_splits <- vfold_cv(ames_train, strata = "Sale_Price", v = 4)

# ------------------------------------------------------------------------------

ames_rec <-
  recipe(Sale_Price ~ ., data = ames_train) %>%
  step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = .1)  %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_ns(Longitude, deg_free = tune("lon")) %>%
  step_ns(Latitude, deg_free = tune("lat"))

knn_model <-
  nearest_neighbor(
    mode = "regression",
    neighbors = tune("K"),
    weight_func = tune(),
    dist_power = tune()
  ) %>%
  set_engine("kknn")

ames_wflow <-
  workflow() %>%
  add_recipe(ames_rec) %>%
  add_model(knn_model)

ames_set <-
  parameters(ames_wflow) %>%
  update(K = neighbors(c(1, 50)))

set.seed(7014)
ames_grid <-
  ames_set %>%
  grid_max_entropy(size = 5)

ames_grid_search <-
  tune_grid(
    ames_wflow,
    resamples = rs_splits,
    grid = ames_grid,
    control = control_grid(verbose = TRUE)
  )

final_ames_wflow <- finalize_workflow(ames_wflow, select_best(ames_grid_search, metric = 'rmse')) %>% 
  fit(data = ames_train)

ames_preds <- final_ames_wflow %>% predict(ames_test) %>% bind_cols(ames_test)

df_preds <- wf_final %>% predict(df_test, type = "prob") %>% bind_cols(df_test)
df_preds <- wf_final %>% predict(df_test) %>% bind_cols(df_preds)

set.seed(2082)
ames_iter_search <-
  tune_bayes(
    ames_wflow,
    resamples = rs_splits,
    param_info = ames_set,
    initial = ames_grid_search,
    iter = 15
  )
