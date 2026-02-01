# ignore, old functions not used in any analysis 


def prscsx_cv_retune(N, geno_prscsx, pheno_prscsx, prscsx_weights, m):
    # NOTE: edit to only tune single pop penalty...not sure why there is an outer loop for penalty_prscsx when there is not second hyperparameter like magepro 
    # N = sample size 
    # geno_prscsx = genotypes
    # pheno_prscsx = gene expression or phenotype data 
    # prscsx_weights = dictionary of external datasets to use from prscsx shrinkage 
        # Ancestry : Array of weights
    # m = str indicating type of model ['enet', 'lasso']

    if m not in ['enet', 'lasso']:
        print("Invalid model type passed to param m")
        sys.exit(1)

    alphas_prscsx=np.logspace(-2,1.2,50,base=10)
    kf_prscsx=KFold(n_splits=5)
    #store r2 from lm for each penalty
    lm_r2_allpenalty=[] 
    for penalty_prscsx in alphas_prscsx:
        #everyfold record predicted expression on testing set
        predicted_expressions = np.zeros(N)
        for train_index_prscsx, test_index_prscsx in kf_prscsx.split(geno_prscsx):
            y_train_prscsx, y_test_prscsx = pheno_prscsx[train_index_prscsx], pheno_prscsx[test_index_prscsx]
            y_train_std_prscsx=(y_train_prscsx-np.mean(y_train_prscsx))/np.std(y_train_prscsx)
            y_test_std_prscsx=(y_test_prscsx-np.mean(y_test_prscsx))/np.std(y_test_prscsx)
            X_train_prscsx, X_test_prscsx = geno_prscsx[train_index_prscsx], geno_prscsx[test_index_prscsx]
            #afronly weights
            #if m == 'lasso':
                #model = lm.Lasso(fit_intercept=True)
            #else:
                #model = lm.ElasticNet(fit_intercept=True)
            #model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
            #model.fit(X_train_prscsx, y_train_std_prscsx)
            #coef_prscsx = model.coef_
            best_penalty_single_tuning, coef_prscsx, single_r2_tuning = fit_sparse_regularized_lm(X_train_prscsx, y_train_std_prscsx, m)
            print("best penalty " + m + " in cross validation iteration: " + str(best_penalty_single_tuning) )
            if np.all(coef_prscsx == 0):
                wgts = weights_marginal_FUSION(X_train_prscsx, y_train_std_prscsx, beta = True)
                coef_prscsx, r2_top1 = top1(y_test_std_prscsx, X_test_prscsx, wgts)
            #prepare for linear regression to find optimal combination of AFR gene model and EUR sumstat
            X_train_prscsx2 = np.dot(X_train_prscsx, coef_prscsx.reshape(-1, 1))
            X_test_prscsx2 = np.dot(X_test_prscsx, coef_prscsx.reshape(-1, 1))
            for ancestry, weights in prscsx_weights.items():
                X_train_prscsx2 = np.hstack((X_train_prscsx2, np.dot(X_train_prscsx, weights.reshape(-1, 1))))
                X_test_prscsx2 = np.hstack((X_test_prscsx2, np.dot(X_test_prscsx, weights.reshape(-1, 1))))
            linear_regression = LinearRegression()
            linear_regression.fit(X_train_prscsx2, y_train_std_prscsx)
            linear_coef = linear_regression.coef_
            #predict on testing
            predicted_expressions[test_index_prscsx] = np.dot(X_test_prscsx2, linear_coef)
        #record r2 from lm
        lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), pheno_prscsx)
        r2_cv_penalty = lreg.score(np.array(predicted_expressions).reshape(-1, 1), pheno_prscsx)
        lm_r2_allpenalty.append(r2_cv_penalty) 
    besti_prscsx=np.argmax(lm_r2_allpenalty)
    bestalpha_prscsx=alphas_prscsx[besti_prscsx]
    prscsx_r2 = lm_r2_allpenalty[besti_prscsx] # returning this for cv r2

    # full model for training r2 and full coef
    #if m == 'lasso':
        #model = lm.Lasso(fit_intercept=True)
    #else:
        #model = lm.ElasticNet(fit_intercept=True)
    #model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
    #model.fit(geno_prscsx, pheno_prscsx)
    #coef_prscsx = model.coef_
    best_penalty_single_tuning_full, coef_prscsx, single_r2_tuning_full = fit_sparse_regularized_lm(geno_prscsx, pheno_prscsx, m)
    if np.all(coef_prscsx == 0):
        wgts = weights_marginal_FUSION(geno_prscsx, pheno_prscsx, beta = True)
        coef_prscsx, r2_top1 = top1(pheno_prscsx, geno_prscsx, wgts)
    X_prscsx = np.dot(geno_prscsx, coef_prscsx.reshape(-1, 1))
    for ancestry, weights in prscsx_weights.items():
        X_prscsx = np.hstack((X_prscsx, np.dot(geno_prscsx, weights.reshape(-1, 1))))
    linear_regression = LinearRegression()
    linear_regression.fit(X_prscsx, pheno_prscsx)
    linear_coef = linear_regression.coef_
    wgts_sep = coef_prscsx.reshape(-1, 1)
    for ancestry, weights in prscsx_weights.items():
        wgts_sep = np.hstack((wgts_sep, weights.reshape(-1, 1)))
    prscsx_coef = np.dot(wgts_sep, linear_coef)

    return prscsx_r2, prscsx_coef


def magepro_cv_retune(N, geno_magepro, pheno_magepro, susie_weights, m):
    ### version of magepro_cv which retunes best alpho for single ancestry model with only the training split. 
    # N = sample size 
    # geno_magepro = genotypes
    # pheno_magepro = gene expression or phenotype data 
    # susie_weights = dictionary of external datasets to use (susie posterior weights)
        # Ancestry : Array of weights
    # m = str indicating type of model ['enet', 'lasso']

    if m not in ['enet', 'lasso']:
        print("Invalid model type passed to param m")
        sys.exit(1)

    alphas_magepro=np.logspace(-2,1.2,50,base=10)
    kf_magepro=KFold(n_splits=5)
    #store r2 from lm for each penalty
    lm_r2_allpenalty=[] 
    for penalty_magepro in alphas_magepro:
        #everyfold record predicted expression on testing set
        predicted_expressions = np.zeros(N)
        for train_index_magepro, test_index_magepro in kf_magepro.split(geno_magepro):
            y_train_magepro, y_test_magepro = pheno_magepro[train_index_magepro], pheno_magepro[test_index_magepro]
            y_train_std_magepro=(y_train_magepro-np.mean(y_train_magepro))/np.std(y_train_magepro)
            y_test_std_magepro=(y_test_magepro-np.mean(y_test_magepro))/np.std(y_test_magepro)
            X_train_magepro, X_test_magepro = geno_magepro[train_index_magepro], geno_magepro[test_index_magepro]
            # afronly weights
            #if m == 'lasso':
                #model = lm.Lasso(fit_intercept=True)
            #else:
                #model = lm.ElasticNet(fit_intercept=True)
            #model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
            #model.fit(X_train_magepro, y_train_std_magepro)
            #coef_magepro = model.coef_
            best_penalty_single_tuning, coef_magepro, single_r2_tuning = fit_sparse_regularized_lm(X_train_magepro, y_train_std_magepro, m)
            print("best penalty " + m + " in cross validation iteration: " + str(best_penalty_single_tuning) )
            if np.all(coef_magepro == 0):
                wgts = weights_marginal_FUSION(X_train_magepro, y_train_std_magepro, beta = True)
                coef_magepro, r2_top1 = top1(y_test_std_magepro, X_test_magepro, wgts)
            #prepare for ridge regression to find optimal combination of AFR gene model and EUR sumstat
            X_train_magepro2 = np.dot(X_train_magepro, coef_magepro.reshape(-1, 1))
            X_test_magepro2 = np.dot(X_test_magepro, coef_magepro.reshape(-1, 1))
            for ancestry, weights in susie_weights.items():
                X_train_magepro2 = np.hstack((X_train_magepro2, np.dot(X_train_magepro, weights.reshape(-1, 1))))
                X_test_magepro2 = np.hstack((X_test_magepro2, np.dot(X_test_magepro, weights.reshape(-1, 1))))
            ridge = Ridge(alpha=penalty_magepro) #now finding best penalty for ridge regression 
            ridge.fit(X_train_magepro2,y_train_std_magepro)
            ridge_coef = ridge.coef_
            #predict on testing
            predicted_expressions[test_index_magepro] = np.dot(X_test_magepro2, ridge_coef)
        #record r2 from lm
        lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), pheno_magepro)
        r2_cv_penalty = lreg.score(np.array(predicted_expressions).reshape(-1, 1), pheno_magepro)
        lm_r2_allpenalty.append(r2_cv_penalty) 
    besti_magepro=np.argmax(lm_r2_allpenalty)
    bestalpha_magepro=alphas_magepro[besti_magepro]
    magepro_r2 = lm_r2_allpenalty[besti_magepro] # returning this for cv r2

    # full model for training r2 and full coef
    #if m == 'lasso':
        #model = lm.Lasso(fit_intercept=True)
    #else:
        #model = lm.ElasticNet(fit_intercept=True)
    #model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
    #model.fit(geno_magepro, pheno_magepro)
    #coef_magepro = model.coef_
    best_penalty_single_tuning_full, coef_magepro, single_r2_tuning_full = fit_sparse_regularized_lm(geno_magepro, pheno_magepro, m)
    if np.all(coef_magepro == 0):
        wgts = weights_marginal_FUSION(geno_magepro, pheno_magepro, beta = True)
        coef_magepro, r2_top1 = top1(pheno_magepro, geno_magepro, wgts)
    X_magepro = np.dot(geno_magepro, coef_magepro.reshape(-1, 1))
    for ancestry, weights in susie_weights.items():
        X_magepro = np.hstack((X_magepro, np.dot(geno_magepro, weights.reshape(-1, 1))))
    ridge = Ridge(alpha=bestalpha_magepro)
    ridge.fit(X_magepro, pheno_magepro)
    wgts_sep = coef_magepro.reshape(-1, 1)
    for ancestry, weights in susie_weights.items():
        wgts_sep = np.hstack((wgts_sep, weights.reshape(-1, 1)))
    magepro_coef = np.dot(wgts_sep, ridge.coef_) #magepro coef

    return magepro_r2, magepro_coef
