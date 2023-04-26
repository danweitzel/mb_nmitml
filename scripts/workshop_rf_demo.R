## Title: MB Workshop Script
## Purpose: Implement a random forest using V-Dem Data
## Date: 2023-04-25
## Updated: 2023-04-26
## Author: Iasmin Goes
##         Daniel Weitzel

# Turn of scientific notation
options(scipen=999)

## Libraries
#devtools::install_github("vdeminstitute/vdemdata", force = TRUE)
library("tidyverse") # For data processing
library("vdemdata")  # The data set we will use
library("h2o")       # the machine learning package 
library("randomizr") # for grouped fold assignment
library("naniar")    # Missing data visualization
library("ggpubr")    # Combine graphs

## Data preprocessing
## This code snippet downloads the most recent V-Dem data from Github and reduces the very large 
## data set to a smaller size. We only keep variables that we will use in our random forest. 
## These variables are ID variables (country and year identifiers), our outcome (Liberal Democracy),
## and a set of objective indicators of electoral democracy.
## We also implement a couple of feature engineering steps, such as generating an indicator for 
## legislative and presidential elections and calculating vote share difference variables. 

df_vdem <- vdemdata::vdem |> 
  dplyr::select(country_name, country_text_id, country_id, year, # country and year identifiers 
                v2x_libdem, # Democracy Indicator
                v2xlg_elecreg, v2xex_elecreg, # legislative and presidential regular elections
                v2ellovtlg, v2ellovtsm, v2ellovttm, # Vote shares legislative elections 
                v2elloseat, v2ellostsl, v2ellostss, v2ellostts, # seat and seat shares legislature 
                v2elvotlrg, v2elvotsml, # presidential elections vote shares 
                v2elmulpar_ord, # legislative elections are multi-party
                v2x_suffr, # share of population with suffrage 
                v2msuffrage, v2fsuffrage, # male and female suffrage 
                v2eltype_0, v2eltype_1, # legislative elections first and second round
                v2eltype_6, v2eltype_7, # presidential elections first and second round
                v2ellocons, v2ellocumul,  # legislative elections consecutive and cummulative 
                v2elprescons, v2elprescumul, # presidential elections consecutive and cummulative 
                v2elturnhog, v2elturnhos, v2eltvrexo, # hog, hos, and executive turnover
                v2svdomaut, # domestic autonomy 
                v2svindep) |> # independent state 
  mutate(v2eltype_legislative = pmax(v2eltype_0, v2eltype_1),
         v2eltype_legislative = ifelse(v2eltype_legislative == 0 , NA, v2eltype_legislative),
         v2eltype_presidential = pmax(v2eltype_6, v2eltype_7),
         v2eltype_presidential = ifelse(v2eltype_presidential == 0 , NA, v2eltype_presidential)) |> 
  dplyr::select(-c(v2eltype_0, v2eltype_1, v2eltype_6, v2eltype_7)) |> 
  group_by(country_text_id) |> 
  fill(c(v2ellovtlg, v2ellovtsm, v2ellovttm,
         v2elloseat, v2ellostsl, v2ellostss, v2ellostts,
         v2elvotlrg, v2elvotsml,
         v2elturnhog, v2elturnhos, v2eltvrexo, 
         v2eltype_legislative,v2eltype_presidential)) |> 
  mutate_at(vars(v2ellovtlg, v2ellovtsm, v2ellovttm, v2elloseat, v2ellostsl, v2ellostss, v2ellostts), ~ ifelse(v2xlg_elecreg == 0, NA, .)) |> 
  mutate_at(vars(v2elvotlrg, v2elvotsml), ~ ifelse(v2xex_elecreg == 0, NA, .)) |> 
  mutate(top2_difference = ifelse(!is.na(v2ellovtlg) & !is.na(v2ellovtsm), v2ellovtlg-v2ellovtsm, 
                                  ifelse(!is.na(v2ellovtlg) & is.na(v2ellovtsm), v2ellovtlg, NA)), 
         top2_combined = ifelse(!is.na(v2ellovtlg) & !is.na(v2ellovtsm), v2ellovtlg+v2ellovtsm, 
                                ifelse(!is.na(v2ellovtlg) & is.na(v2ellovtsm), v2ellovtlg, NA)), 
         top2_monopoly = ifelse(top2_combined > 59.99, 1, 0)) |> 
  dplyr::filter(year > 1900) |> 
  dplyr::filter(year < 2021) |> 
  ungroup()

## Plotting the data 
## Inspecting the dependent variable 
ggplot(df_vdem, aes(x = v2x_libdem)) +
  geom_histogram() +
  theme_bw() +
  labs(title = "Distribution of the dependent variable",
       y = "Count",
       x = "V-Dem's Liberal Democracy Index")

## Missing data in the data set
gg_miss_fct(df_vdem, year) +
  theme_minimal() + labs(y = "Variable", x = "Year") +
  theme(legend.position = "bottom")

## Machine learning model
## The outcome we want to explain
outcome <- "v2x_libdem"

## A set of identifiers we want to have in the data but not use in the model
## Using these variables in the model would be highly problematic, the algorithm would learn 
## what democracy values are associated with specific countries.
ids         <- c("country_name", "country_text_id", "country_id", "year")

## The variables we want to use as predictors in our random forest model 
## This is a list of objective indicators from V-Dem. We can use them to train a machine learning model 
## on the subjective indicator of liberal democracy
preds       <- c("v2xlg_elecreg", "v2xex_elecreg", # legislative and presidential regular elections
                 "v2ellovtlg", "v2ellovtsm", "v2ellovttm", # Vote shares legislative elections 
                 "v2ellostsl", "v2ellostss", "v2ellostts", # seat and seat shares legislature 
                 "v2elvotlrg", "v2elvotsml", # presidential elections vote shares 
                 "v2elmulpar_ord", # legislative elections are multi-party
                 "v2x_suffr", # share of population with suffrage 
                 "top2_difference", "top2_combined", "top2_monopoly",
                 "v2elturnhog", "v2elturnhos", "v2eltvrexo", # hog, hos, and executive turnover
                 "v2svindep")

## Generate a training data set
## In this scenario we train the model on the years between 1900 and 2011
## The test data set will be the years after 2011
df_vdem_train <- 
  df_vdem |>
  dplyr::select(all_of(ids), all_of(preds), all_of(outcome)) |>
  drop_na(all_of(outcome)) |>
  dplyr::filter(year <= 2011) 

## Generate a six fold cross-validation data set that groups entire countries into folds 
## This allows us to train the model in a way to optimize it for out of sample prediction of 
## entire countries
df_vdem_train$folds <- cluster_ra(clusters = df_vdem_train$country_id, 
                                  conditions = c("Fold_1", "Fold_2", "Fold_3", "Fold_4", 
                                                 "Fold_5", "Fold_6")) 

## For easier processing later on: generate a data set with the IDs of the training data set
df_vdem_train_ids <-
  df_vdem_train |>
  dplyr::select(all_of(ids))

## We now remove the IDs from the training data set so the random forest can't use them 
## to learn 
df_vdem_train <-
  df_vdem_train |>
  dplyr::select(-c(all_of(ids)))

## Generate a test data set
## The test data set is all years after 2011
df_vdem_predict <- 
  df_vdem |>
  filter(year > 2011) |> 
  dplyr::select(all_of(ids), all_of(preds), all_of(outcome))


## For easier processing later on: generate a data set with the IDs of the test data set
df_vdem_predict_ids <-
  df_vdem_predict |>
  dplyr::select(all_of(ids))

## We now remove the IDs from the test data set so the random forest can't use them 
## to learn 
df_vdem_predict <-
  df_vdem_predict |>
  dplyr::select(-c( all_of(ids)))


## H20
# Random Forest implementation with h20
# configuration uses all cores and 20GB of RAM 
h2o.no_progress()
h2o.init(nthreads=-1, max_mem_size = "20g")
h2o.removeAll()

## set the predictor names
predictors <- setdiff(colnames(df_vdem_train), outcome)
train_h2o  <- as.h2o(df_vdem_train)
test_h2o   <- as.h2o(df_vdem_predict)

# Estimate the random forest model
model_rf_libdem <- 
  h2o.randomForest(
    model_id = "vdem_ld_1",         # id number of the model 
    x = predictors,                 # the predictors we want to use
    y = outcome,                    # the outcome we want to use
    fold_column = "folds",          # here we specify the name of the xval fold column
    training_frame = train_h2o,     # the name of the training data set
    ntrees = 400,                   # the number of trees to generate
    mtries = 4,                     # numbers of columsn to select at each level
    col_sample_rate_per_tree = 0.8, # sample rate of columns at each level
    seed = 1904,                    # the seed 
    keep_cross_validation_predictions = TRUE)

## Examining the performance of the model
h2o.r2(model_rf_libdem, train = TRUE, xval = TRUE)
h2o.performance(model_rf_libdem, train = TRUE)
h2o.performance(model_rf_libdem, xval = TRUE)


## Generating a nice looking variable importance plot
var_importance_tibble_rf1 <- 
  as_tibble(h2o.varimp(model_rf_libdem)) %>%
  select(variable,scaled_importance) %>%
  filter(scaled_importance > 0.05) %>% # only top 10
  mutate(variable = case_when(
    variable == "v2ellovtlg" ~ "Lower chamber Vote Share, largest party",
    variable == "v2ellovtsm" ~ "Lower chamber Vote Share, second largest party",
    variable == "v2ellovttm" ~ "Lower chamber Vote Share, third largest party",
    variable =="v2elloseat" ~ "Lower chamber election seats",
    variable == "v2ellostsl" ~ "Lower chamber Seat Share, largest party",
    variable == "v2ellostss" ~ "Lower chamber Seat Share, second largest party",
    variable == "v2ellostts" ~ "Lower chamber Seat Share, third largest party",
    variable == "v2elvotlrg" ~ "Presidential Vote Share, largest party",
    variable == "v2elvotsml" ~ "Presidential Vote Share, second largest party",
    variable == "v2xlg_elecreg" ~ "Electoral Regime Index, Lower chamber",
    variable == "v2xex_elecreg" ~ "Electoral Regime Index, Presidential",
    variable == "v2x_suffr" ~ "Share of population with suffrage ",
    variable == "v2msuffrage" ~ "Share of male population with suffrage ",
    variable == "v2fsuffrage" ~ "Share of female population with suffrage ",
    variable == "v2elmulpar_ord" ~ "Elections multiparty (Ordinal)",
    variable == "v2eltype_legislative" ~ "Lower chamber Election",
    variable == "v2eltype_presidential" ~ "Presidential Election",
    variable == "v2elprescons" ~ "Presidential elections consecutive ",
    variable == "v2elprescumul" ~ "Presidential elections cumulative",
    variable == "v2ellocons" ~ "Lower chamber election consecutive",
    variable == "v2ellocumul" ~ "Lower chamber election cumulative",
    variable == "v2elturnhog" ~ "Election HOG turnover ordinal",
    variable == "v2elturnhos" ~ "Election HOS turnover ordinal",
    variable == "v2eltvrexo" ~ "Election executive turnover ordinal",
    variable == "v2svdomaut" ~ "Domestic autonomy ",
    variable == "top2_combined" ~ "Vote Share, two largest parties",
    variable == "top2_difference" ~ "Difference Vote Share, two largest parties",
    variable == "top2_monopoly" ~ "Vote Share Top2 combined >= 60%",
    variable == "v2svindep" ~ "Independent states"))

#  VIP plot
ggplot(var_importance_tibble_rf1, aes(x = reorder(variable, scaled_importance), y = scaled_importance)) + geom_col() +
  theme_minimal() + coord_flip() + labs(x = " ", y = "Scaled Importance") + 
  theme(axis.text.y = element_text(size = 12), axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

## Add predictions to the training data set 
preds_training    <- h2o.predict(object = model_rf_libdem, newdata = train_h2o)
df_vdem_train$preds_training <- as.vector(preds_training)

## Add predictions to the test data set 
preds_test        <- h2o.predict(object = model_rf_libdem, newdata = test_h2o)
df_vdem_predict$preds_test   <- as.vector(preds_test)

## Combine the training and test data set  
df_combined <-  
  df_vdem_train |> 
  bind_cols(df_vdem_train_ids) |> 
  bind_rows(df_vdem_predict|> 
              bind_cols(df_vdem_predict_ids)) |> 
  unite("country_year", c("country_text_id", "year"), sep = " ", remove = FALSE) %>% 
  mutate(preds_value = ifelse(!is.na(preds_training), preds_training,
                              ifelse(!is.na(preds_test), preds_test, NA)),
         preds_type = ifelse(!is.na(preds_training), "training",
                             ifelse(!is.na(preds_test), "test", NA)))

## Visualize the predictions
## First two scatterplots that show the observed vs predicted in train and test set
fig_pred1 <- 
  df_combined %>% 
  filter(preds_type %in% c("training")) %>% 
  ggplot(aes(x = v2x_libdem, y = preds_value)) +
  geom_point(size=0.1, color = "darkgray") + theme_minimal() + 
  xlim(0,1) + ylim(0,1) +
  geom_abline(intercept = 0, slope = 1, color = "gray26") +
  labs(title = "Prediction on Training Dataset", subtitle = "N = 16,911, 91% Subset of available V-Dem Data",
       caption = "Note: Training on data from 1900 to 2010. R2 test 0.951, xval 0.746",
       x = "Liberal Democracy (observed)", y = "Liberal Democracy (predicted)") +
  geom_text(label = df_combined$country_year[df_combined$preds_type == "training"], 
            check_overlap = T)

fig_pred2 <- 
  df_combined %>% 
  filter(preds_type %in% c("test")) %>% 
  ggplot(aes(x = v2x_libdem, y = preds_value)) +
  geom_point(size=0.1, color = "darkgray") + theme_minimal() + 
  xlim(0,1) + ylim(0,1) +
  geom_abline(intercept = 0, slope = 1, color = "gray26")  +
  labs(title = "Prediction on Test Dataset", subtitle = "N = 1,611, 9% Subset of available V-Dem Data",
       caption = "Note: Prediction on data since 2010.",
       x = "Liberal Democracy (observed)", y = "Liberal Democracy (predicted)") +
  geom_text(label = df_combined$country_year[df_combined$preds_type == "test"], 
            check_overlap = T)
# Combine the graph
ggarrange(fig_pred1, fig_pred2, 
          labels = c("A", "B"),
          ncol = 2, nrow = 1)

## Next visualize the predictions in the test set vs the observed values for 
## select countries using a line graph
df_combined |>
  filter(country_name %in% c("United States of America", "Germany",
                             "France", "Russia", "Austria", "Colombia",
                             "Mexico", "Brazil", "Argentina")) |>
  filter(year > 2000) |> 
  ggplot(aes(x = year)) +
  geom_line(aes(y = v2x_libdem)) +
  geom_line(aes(y = preds_test), color = "red") +
  facet_wrap(~country_name) + ylim(0,1) +
  theme_bw() +
  labs(title = "Comparing observed and predicted values",
       x = "Year",
       y = "V-Dem's Liberal Democracy Indicator")

# fin
