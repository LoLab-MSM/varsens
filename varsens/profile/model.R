# This code fits the data to a statistical model that is predictive
# of the accuracy of Saltelli's method. Thus, one can find a necessary
# n value to achieve the desire accuracy

setwd("~/Projects/varsens/") # Varsens project directory
setwd("varsens/profile/")    # Specific subdirectory in project


# This data is also for models where the dimensions are independent. What this
# says about dependent dimensions is not evaluated. I suspect that under
# dependency the accuracy improves, but this has not been tested.

#################################
# Data Loading
d6   <- read.csv('error-profile-dim6.csv'  , header=FALSE)  #   6 dimensional models
d12  <- read.csv('error-profile-dim12.csv' , header=FALSE)  #  12 dimensional models
d24  <- read.csv('error-profile-dim24.csv' , header=FALSE)  #  24 dimensional models
d48  <- read.csv('error-profile-dim48.csv' , header=FALSE)  #  48 dimensional models
d96  <- read.csv('error-profile-dim96.csv' , header=FALSE)  #  96 dimensional models
d192 <- read.csv('error-profile-dim192.csv', header=FALSE)  # 192 dimensional models

d6$dim   <-   6
d12$dim  <-  12
d24$dim  <-  24
d48$dim  <-  48
d96$dim  <-  96
d192$dim <- 196

data <- rbind(d6, d12, d24, d48, d96, d192)
colnames(data) <- c("n", "mu", "sd", "lci", "uci", "max", "dim")

#################################
# Transforms
data$log10N   <- log10(data$n)
data$log10sqrtMax <- log10(sqrt(data$max))  # Max is maximum observed error under 30 random runs
data$log10dim <- log10(data$dim)

model1 <- lm(log10sqrtMax ~ log10N*log10dim, data)
model2 <- lm(log10sqrtMax ~ log10N*dim, data)
summary(model1)
summary(model2)

data$interact <- data$dim * data$log10N
model3 <- lm(log10sqrtMax ~ log10N + interact, data)

#################################
# Plot Results
plot(data$log10N, data$log10sqrtMax, pch = as.numeric(factor(data$dim)), xlab="Log10(n)", ylab="maximum error")
legend(3.75, 0.2, c("6", "12", "24", "48", "96", "192"), pch=1:6, lty=1:6)

params <- coef(model3)
abline(params[1], params[2]+params[3]*6,   lty=1)
abline(params[1], params[2]+params[3]*12,  lty=2)
abline(params[1], params[2]+params[3]*24,  lty=3)
abline(params[1], params[2]+params[3]*48,  lty=4)
abline(params[1], params[2]+params[3]*96,  lty=5)
abline(params[1], params[2]+params[3]*192, lty=6)

# So for 197, n=10000, the predicted error is
# 
n <- 10000
d <- 197
10^(params[1] + (params[2] + params[3]*d)*log10(n) )

# Predicted accuracy at this is  0.02657896 . So anything below this is not trustable for an earm run.

n <- 10000
d <- 106
10^(params[1] + (params[2] + params[3]*d)*log10(n) )
