def analyze_naive_bayes(datasets):
    results = {}

    for name, X in datasets.items():
        y = X['class']
        X = np.array(X.drop(columns=['class'], axis=1))
        n_features = X.shape[1]
        feature_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

        for (i, j) in feature_pairs:
            # Extract feature pair
            X_pair = X[:, [i, j]]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_pair, y, test_size=0.3, random_state=42)

            # Train Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            #y_pred = model.predict(X_test)

            # Calculate ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Sensitivity-Specificity point
            idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[idx]
            best_sen = tpr[idx]
            best_spe = 1 - fpr[idx]

            # Bootstrap confidence intervals
            n_bootstrap = 1000
            aucs = []
            tprs = []
            for _ in range(n_bootstrap):
                X_resampled, y_resampled = resample(X_test, y_test)
                y_prob_resampled = model.predict_proba(X_resampled)[:, 1]
                fpr_resampled, tpr_resampled, _ = roc_curve(y_resampled, y_prob_resampled)
                roc_auc_resampled = auc(fpr_resampled, tpr_resampled)
                aucs.append(roc_auc_resampled)
                tprs.append(np.interp(fpr, fpr_resampled, tpr_resampled))

            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            z_score = norm.ppf(0.975)  # For 95% CI
            confidence_lower = max(0, mean_auc - z_score * std_auc)
            confidence_upper = min(1, mean_auc + z_score * std_auc)

            # Visualization
            plt.figure(figsize=(12, 6))

            # # Plot feature space
            # plt.subplot(1, 2, 1)
            # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, cmap='coolwarm', label='Test Data')
            # plt.xlabel(f'Feature {i + 1}')
            # plt.ylabel(f'Feature {j + 1}')
            # plt.title(f'{name}: Feature {i + 1} vs Feature {j + 1}')
            # plt.legend()

            plt.subplot(1, 2, 1)
            cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
            cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

            # Разделение
            x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
            y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap=cmap_bold)
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel(f'Feature {j + 1}')
            plt.title(f'{name}: Feature {i + 1} vs Feature {j + 1}')

            # KDE
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_pair)
            log_density = kde.score_samples(np.c_[xx.ravel(), yy.ravel()])
            density = np.exp(log_density).reshape(xx.shape)
            plt.contour(xx, yy, density, levels=10, cmap='coolwarm', alpha=0.6)

            # ROC-curve
            plt.subplot(1, 2, 2)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
            plt.fill_between(fpr, 
                             np.maximum(0, np.mean(tprs, axis=0) - z_score * np.std(tprs, axis=0)), 
                             np.minimum(1, np.mean(tprs, axis=0) + z_score * np.std(tprs, axis=0)), 
                             color='blue', alpha=0.2, label='95% CI')
            plt.scatter(1 - best_spe, best_sen, color='red', zorder=10, label='Best Threshold')
            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'ROC Curve with 95% CI')
            plt.legend()

            plt.tight_layout()
            plt.show()

            # # Save results
            # results[f'{name}_f{i+1}_f{j+1}'] = {
            #     'roc_auc': roc_auc,
            #     'confidence_interval': (confidence_lower, confidence_upper),
            #     'best_threshold': best_threshold,
            #     'best_sensitivity': best_sen,
            #     'best_specificity': best_spe
            # }

    # return results