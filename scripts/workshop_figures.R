library(tidyverse)
library(scales)
library(ggpubr)

# Load the data
df_income <- read_csv("~/Dropbox/a_preco_de_banana/Machine_Learning/Workshop/data/Income1.csv") |> 
  dplyr::select(-...1)

# Linear regression
fit_lm <- lm(Income ~ Education, data = df_income)
df_income$lm_predicted <- predict(fit_lm)   
df_income$lm_residuals <- residuals(fit_lm)

# Loess regression
fit_lm75 <- loess(Income ~ Education, data=df_income, span=.75)
df_income$loess_predicted <- predict(fit_lm75)   
df_income$loess_residuals <- residuals(fit_lm75)

# Scatterplot
p1 <-
  ggplot(df_income, aes(x = Education, y = Income)) +
  geom_point() +
  theme_bw()  +
  labs(title = "Scatterplot of Income and Education") + scale_y_continuous( breaks=pretty_breaks()) +
  scale_y_continuous( breaks=pretty_breaks()) +
  scale_x_continuous( breaks=pretty_breaks())


p2 <- 
  ggplot(df_income, aes(x = Education, y = Income)) +
  geom_smooth(method = "loess", se = FALSE, color = "forestgreen")  +
  geom_point() + theme_bw()  + 
  geom_point(aes(y = loess_predicted), shape = 1) +  
  geom_segment(aes(xend = Education, yend = loess_predicted), alpha = .7, color = "goldenrod3")  +
  labs(title = "Loess regression") +
  scale_y_continuous( breaks=pretty_breaks()) +
  scale_x_continuous( breaks=pretty_breaks())


p3 <- 
  ggplot(df_income, aes(x = Education, y = Income)) +
  geom_smooth(method = "lm", se = FALSE, color = "forestgreen")  +
  geom_point() + theme_bw()  + 
  geom_point(aes(y = lm_predicted), shape = 1) +  
  geom_segment(aes(xend = Education, yend = lm_predicted), alpha = .7, color = "goldenrod3") +
  labs(title = "Linear regression") +
  scale_y_continuous( breaks=pretty_breaks()) +
  scale_x_continuous( breaks=pretty_breaks())

ggarrange(p1, p2,p3, ncol=3)


