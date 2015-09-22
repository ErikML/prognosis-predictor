library(glmnet)

summary <- read.csv("DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv")
reimb <- summary$MEDREIMB_IP + summary$BENRES_IP + summary$PPPYMT_IP + summary$MEDREIMB_OP + 
         summary$BENRES_OP + summary$PPPYMT_OP + summary$MEDREIMB_CAR + summary$BENRES_CAR + 
         summary$PPPYMT_CAR
reimb[reimb < 0 ] <- 0
log_reimb <- log(reimb + 1)
hist(log_reimb, main="Log Reimbursments in 2008")

cond <- summary[,c("BENE_ESRD_IND", "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
                   "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA",
                   "SP_STRKETIA")]
cond$BENE_ESRD_IND <- ifelse(cond$BENE_ESRD_IND == "Y", 1, 2)
cond <- -1 * cond + 2
age <- 2009 - summary$BENE_BIRTH_DT %/% 10000
gender <- -1 * summary$BENE_SEX_IDENT_CD + 2
expensivestate <- ifelse(summary$SP_STATE_CODE %in% c(47, 41, 22, 38, 31, 5, 2, 33, 7, 12),  1, 0)
data = cbind(age, gender, expensivestate, cond)
sub <- sample(nrow(data), floor(nrow(data) * 0.7))
Xtrain <- data[sub,]
Xtest <- data[-sub,]
Ytrain <- log_reimb[sub]
Ytest <- log_reimb[-sub]

input <- cbind(Xtrain, Ytrain)
linear_model <- lm(Ytrain ~ age + gender + expensivestate + BENE_ESRD_IND + SP_ALZHDMTA + SP_CHF + 
                SP_CHRNKIDN + SP_CNCR + SP_COPD + SP_DEPRESSN + SP_DIABETES + SP_ISCHMCHT +
                SP_OSTEOPRS + SP_RA_OA + SP_STRKETIA, data=input)
lm_results <- predict(linear_model, newdata=Xtest)
lm_rmse <- sqrt(mean((lm_results - Ytest)^2))

ridge_model <- glmnet(as.matrix(Xtrain), as.matrix(Ytrain))
min_lambda <- ridge_model$lambda.min
ridge_results <- predict(ridge_model, as.matrix(Xtest), s=min_lambda)
ridge_rmse <- sqrt(mean((ridge_results - Ytest)^2))

baseline_rmse <- sqrt(mean((mean(log_reimb) - log_reimb)^2))

