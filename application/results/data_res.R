########################################
# Data processing
########################################

# load data
YOUR_PATH = "YOUR_PATH"
setwd(YOUR_PATH)
par(mfrow=c(1,1))
filename = "beePA.RDS"
data = readRDS(filename)
dim(data)
names(data)

set.seed(2021)

# get unique records and assign 0 to y if survival < 0.5
uniqrow = unique(data[, c(12, 26)])
fulld = data[, c(12, 26)]
survi = rep(NA, dim(uniqrow)[1])
for (i in 1:257) {
  indi = fulld[, 1] %in% uniqrow[i,1] & fulld[, 2] %in% uniqrow[i,2]
  if (mean(data[indi, 3]) > 0.8){
    survi[i] = 1
  } else if (  mean(data[indi, 3]) < 0.6){
    survi[i] = 0
  } else {
    survi[i] = NA
  }
}


tmp = as.numeric(rownames(uniqrow))
tmpi = as.numeric(rownames(data)) %in% tmp
data = cbind.data.frame(survi, data[tmpi,])
names(data)

hist(survi)

data = data[!is.na(survi), ]
hist(data$col_nov, 50)

dim(data)
summary(data)
names(data)
data = data[, -c(2, 4)]
colnames(data)[1] = "survival"

pairs(data[, c(1, 3:10)])
pairs(data[, c(1, 11:20)])

y = data$survival
hist(y)
n = length(y)
str(data)
data$spring_yr = as.factor(data$spring_yr)

set.seed(2021)
n = 100
indi = sample(length(y), n)
  indtest = !(1:length(y) %in% indi)
  sum(indtest)
  datatest = data[indtest, ]
  save(datatest, file = "beePAtest.RData")
y = y[indi]
data = data[indi, ]
hist(y)

# install.packages("quantable")
require("quantable")
names(data)
XX = robustscale(data[, c(3:33, 35)])
XX = XX$data
data[, c(3:33, 35)] = XX
summary(data)


# remove strong correlation for continuou features
dat = data
if(!require(corrplot)){install.packages("corrplot"); library(corrplot)}
corrplot(cor(data[,sapply(dat, is.numeric)], use="pairwise.complete.obs"), 
         type="upper", diag = T,
         tl.cex = 0.8)
require(caret)
df1 = data[,sapply(dat, is.numeric)]
df2 = cor(df1)
hc = findCorrelation(df2, cutoff=0.7)
hc = sort(hc)
reduced_Data = df1[,-c(hc)]
summary(reduced_Data)
X = reduced_Data
colnames(X)
colnames(data)
corrplot(cor(X, use="pairwise.complete.obs"), 
         type="upper", diag = T,
         tl.cex = 0.8)
names(data)
names(X)

# create dummies
tmp = data[, c(2, 34)]
tmp = model.matrix(~ ., data=tmp,
                   contrasts.arg = lapply(tmp[,sapply(tmp, is.factor)], contrasts, contrasts=FALSE))
colnames(tmp)
tmp = tmp[,-c(1, 4, 7)]
colnames(tmp)
data = cbind(X, tmp)
names(data)
summary(data)
indf = names(data) %in% c("spring_yr2017", "spring_yr2018")
data = data[, !indf]
dim(data)
names(data)
save(data, file = "beePAtrain.RData")


########################################
# Analysis
########################################

# load data
setwd(YOUR_PATH)

load("beePAtrain.RData")
y = data$survival
X = data[, 2:ncol(data)]

load("beePAtest.RData")
# create dummies
datatest[, c(2)] = as.factor(datatest[, c(2)])
tmp = datatest[, c(2, 34)]
tmp = model.matrix(~ ., data=tmp,
                   contrasts.arg = lapply(tmp[,sapply(tmp, is.factor)], contrasts, contrasts=FALSE))
colnames(tmp)
tmp = tmp[,-c(1, 4, 7)]
colnames(tmp)
names(datatest) 
datatest = datatest[, -c(2, 34)]
datatest = cbind(datatest, tmp)
names(datatest)
indf = names(datatest) %in% names(data) 
datatest = datatest[, indf]
names(datatest)
dim(datatest)
yt = datatest$survival
Xt = datatest[, 2:ncol(datatest)]
# robust scale
require("quantable")
names(Xt)
tmp = Xt[, 19:22]
Xt = robustscale(Xt[, 1:18])
Xt = cbind.data.frame(Xt$data, tmp)


setwd(paste0(YOUR_PATH, "/results"))
flb = list.files()[grepl("solb.csv", list.files())]
flb = c(flb[2:length(flb)], flb[1])
flw = list.files()[grepl("solw.csv", list.files())]
flw = c(flw[2:length(flw)], flw[1])

solb = list()
solw = list()
for (i in 1:length(flb)) {
  kn = strsplit(flb, "-")[[i]][1]
  kp = strsplit(flb, "-")[[i]][2]
  
  solb[[i]] = read.csv(flb[i])
  solw[[i]] = read.csv(flw[i])
}
solb
solw


##############################################  
# Fig. 2: prediction error
##############################################

set.seed(2021)
y = data$survival
X = data[, 2:ncol(data)]
ccglmnet = matrix(NA, dim(X)[2]+1, length(solb))
for (i in 1:length(solb)) {
  require(glmnet)
  fitCV = cv.glmnet(as.matrix(X), y, family="binomial", nfolds = 10, standardize = F, alpha=1, type.logistic="modified.Newton")
  plot(fitCV)
  ccglmnet[, i] = as.matrix(coef(fitCV, s = "lambda.min"))
}
colSums(ccglmnet!=0)

nest = 4
solpred = array(NA, c(11, length(solb), nest))
for (i in 1:length(solb)) {
  solbb = cbind.data.frame(solb[[i]], glmnet = ccglmnet[, i])
  solww = cbind.data.frame(solw[[i]], glmnet = rep(TRUE, dim(data)[1]))
  for (j in 1:nest) {
    indb = solbb[ , j] != 0
    indb = indb[2:length(indb)]
    indw = solww[ , j] != 0
    
    y = data$survival
    X = data[, 2:ncol(data)]
    y = y[indw]
    X = X[indw, indb]
    dff = cbind.data.frame(y, X)  
    model <- glm(y ~ ., family=binomial, data = dff)
    summary(model)
    
    # predictions 
    if (sum(indb) == 1) {
      predyt = predict(model, data.frame(X = Xt[, indb]), type="response")
    } else {
      predyt = predict(model, Xt[, indb], type="response")
    }
    hist(predyt, 50)
    indpdy = as.numeric(predyt > 0.5)
    indpdy = as.numeric(predyt > 0.5)
    indpdy = as.factor(c(indpdy, 0, 1))
    indpdy = indpdy[1:(length(indpdy)-2)]
    require(caret)
    cm <- confusionMatrix(indpdy, reference = as.factor(yt))
    solpred[, i, j] = as.matrix(cm$byClass)
  }    
}
rownames(solpred) = names(cm$byClass)
ranc = 3:10
colnames(solpred) = ranc
solpred  
dim(solpred)

indres = 11
dd = cbind.data.frame(value = solpred[indres,,2], Estimator = rep("MIProb", length(solb)))
dd = rbind.data.frame(dd, cbind.data.frame(value = solpred[indres,,3], Estimator = rep("MIP", length(solb))))
dd = rbind.data.frame(dd, cbind.data.frame(value =  rep(mean(solpred[indres,,1]), length(solb)), 
                                           Estimator = rep("enetLTS", length(solb))))
dd = rbind.data.frame(dd, cbind.data.frame(value =  rep(mean(solpred[indres,,4]), length(solb)), 
                                           Estimator = rep("glmnet", length(solb))))
dd$rang = rep(3:10, 4)

# Fig. 2
p = ggplot(dd, aes(x = rang, y = value, color = Estimator, group = Estimator))  +
  geom_line(aes(color = Estimator, linetype = Estimator), size = 1.5, alpha=1) +
  scale_linetype_manual(values=c("solid", "solid", "dashed", "dashed")) +
  ylab("Balanced Accuracy") +
  xlab("Sparsity Level") +
  theme_bw() +
  theme(legend.position = c(0.1, 0.2), legend.title = element_text(size=12),
        legend.text = element_text(size=12)) +
  theme(panel.border = element_rect(colour = "black", fill=NA),
        axis.text = element_text(colour = 1, size = 5),
        legend.background = element_rect(linetype = 1, size = 0.5, colour = 1)) +
  theme(axis.title.x = element_text(size = 12,face="bold"),
        axis.title.y = element_text(size = 12,face="bold")) +
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12)) +
  scale_x_continuous(breaks = 3:10, limits=c(3,10)) +
  coord_cartesian(ylim = c(0.45, 0.6), expand = FALSE, clip = "off") 
p
# 869 x 470


##############################################  
# Tab. 2: estimation
##############################################

y = data$survival
X = data[, 2:ncol(data)]
tmp = solb[[6]]
rownames(tmp) = c("int", colnames(X))
rownames(tmp)[20:23] = c("exp 1-2", "exp 2-5", "exp <1 ", "exp >10")
rownames(tmp)[1] = "interc."
rownames(tmp)[5] = "dd rain"
rownames(tmp)[13] = "sol rad"
rownames(tmp)[16] = "TWI"
rownames(tmp)[19] = "col nov"
require(xtable)
xtable(cbind.data.frame(Variable=rownames(tmp), Description=rownames(tmp)))

require(glmnet)
set.seed(2021)
fitCV = glmnet(as.matrix(X), y, family="binomial")
plot(fitCV)
fitCV = cv.glmnet(as.matrix(X), y, family="binomial", nfolds = 10, standardize = F, alpha=1, type.logistic="modified.Newton")
plot(fitCV)
ccglmnet = as.matrix(coef(fitCV, s = "lambda.min"))
rownames(ccglmnet) = rownames(tmp)

tmp = cbind.data.frame(ccglmnet, tmp[, c(3, 1, 2)])
names(tmp)[1] = "glmnet"

tmp = as.matrix(tmp)
indN = tmp < 0
indP = tmp > 0

tmp[tmp == 0] = ""
tmp[indN] = "red!25"
tmp[indP] = "green!25"

tmp1 = tmp[1:12, ]
tmp2 = tmp[13:23, ]

xtable(t(tmp1), align = c("l", rep("c", 12)))
xtable(t(tmp2), align = c("l", rep("c", 11)))


##############################################  
# Fig. 3: residuals vs residuals
##############################################

# final fit
i=6
# MIP solution
dffull = cbind.data.frame(y, X[, solb[[i]][ , 2] != 0])
modelfull <- glm(y ~ ., family=binomial, data = dffull)
resfull = residuals(modelfull,  type = "pearson")
# MIProb solution
indb = solb[[i]][ , 2] != 0
indb = indb[2:length(indb)]
indw = solw[[i]][ , 2] != 0
y = data$survival
X = data[, 2:ncol(data)]
X = X[, indb]
yout = y[indw==0]
y = y[indw]
Xout = X[indw==0, ]
X = X[indw, ]
dff = cbind.data.frame(y, X)  
model <- glm(y ~ ., family=binomial, data = dff)
summary(model)
plot(residuals(model,  type = "deviance"))
prout = predict(model,  Xout, type="response")

resrob = rep(NA, length(data$survival))
resrob[indw] = residuals(model, "pearson")
resrob[!indw] = (yout-prout)/sqrt(prout*(1-prout))

indww = indw
indww[indww==F] = "Outlying"
indww[indww==T] = "Non-outlying  "
qn1 =  qnorm(0.0125)
qn2 =  qnorm(0.9875)
respl = cbind.data.frame(resfull, resrob, Units=indww)
p = ggplot(respl, aes(x=resfull, y=resrob, color=Units)) +
  geom_point(size=3) +
  scale_color_manual(values = c("black", "red")) +
  geom_hline(yintercept = qn1, color = "red", linetype = "twodash") + 
  geom_hline(yintercept = qn2, color = "red", linetype = "twodash") + 
  ylab("Pearson residuals MIProb") +
  xlab("Pearson residuals MIP") + 
  theme_bw() +
  theme(legend.position = c(0.1, 0.8), legend.title = element_text(size=12), 
        legend.text = element_text(size=12)) + 
  theme(panel.border = element_rect(colour = "black", fill=NA),
        axis.text = element_text(colour = 1, size = 5),
        legend.background = element_rect(linetype = 1, size = 0.5, colour = 1),
        legend.key.height=unit(1, "cm"), 
        legend.key.width=unit(1, "cm")) +
  guides(shape = guide_legend(override.aes = list(size = 3))) +
  theme(axis.title.x = element_text(size = 12,face="bold"),
        axis.title.y = element_text(size = 12,face="bold")) +
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12)) +
  theme(plot.margin=grid::unit(c(0,0,0,0), "mm")) + 
  coord_cartesian(ylim = c(-15, 25), xlim = c(-2.5, 2.5), expand = FALSE, clip = "off") 
p 
# 869 x 470


# plot(1:100, c(residuals(model, "pearson"), (yout-prout)/sqrt(prout*(1-prout))))
plot(1:100, c(residuals(model, "pearson"), (yout-prout)/sqrt(prout*(1-prout))))


##############################################  
# Fig. 4: outliers box plots
##############################################

out = data[solw[[i]][,2]==0, ]
nout = data[solw[[i]][,2]==1, ]

ddd = cbind.data.frame(data, "Units" = solw[[i]][,2])
inddd = colnames(ddd) %in% c(colnames(X), "Units") 
ddd = ddd[, inddd]
ddd$Units[ddd$Units == 0] = "Outliers"
ddd$Units[ddd$Units == 1] = "Non-outlying  "

# Reshaping data
require(reshape2)
ddl <- melt(ddd, id = "Units")
head(ddl) 
ddl$Units = as.factor(ddl$Units)
p1 = ggplot(ddl, aes(x = variable, y = value, color=Units, fill=Units)) +  # ggplot function
  geom_boxplot(lwd=0.5, width=0.7, alpha=1,
               outlier.size = 3, outlier.alpha = 1, outlier.shape = 21, color="black") +
  ylab("Value") +
  xlab("Variable") +
  theme_bw() +
  theme(legend.position = c(0.1, 0.8), legend.title = element_text(size=12), 
        legend.text = element_text(size=12)) + 
  theme(panel.border = element_rect(colour = "black", fill=NA),
        axis.text = element_text(colour = 1, size = 5),
        legend.background = element_rect(linetype = 1, size = 0.5, colour = 1),
        legend.key.height=unit(1, "cm"), 
        legend.key.width=unit(1, "cm")) +
  guides(shape = guide_legend(override.aes = list(size = 3))) +
  theme(axis.title.x = element_text(size = 12,face="bold"),
        axis.title.y = element_text(size = 12,face="bold")) +
  theme(axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12)) +
  theme(plot.margin=grid::unit(c(0,0,0,0), "mm")) # +
p1 
# 869 x 470

