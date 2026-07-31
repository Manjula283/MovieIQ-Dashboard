import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import ttest_ind, pearsonr, chi2_contingency

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="MovieIQ Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 MovieIQ Dashboard")
st.markdown("### Predictive Analytics on Film Success")

st.sidebar.header("About")
st.sidebar.info(
    """
    **MovieIQ Dashboard**

    This dashboard performs:

    ✅ Data Cleaning

    ✅ Exploratory Data Analysis (EDA)

    ✅ Statistical Analysis

    ✅ Machine Learning

    ✅ Movie Success Prediction
    """
)

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your movies.csv file",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # -------------------------------
    # REQUIRED COLUMNS
    # -------------------------------
    required_columns = [
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average",
    ]

    if not all(col in df.columns for col in required_columns):
        st.error(
            "Dataset must contain:\n"
            "budget, revenue, popularity, runtime, vote_average"
        )
        st.stop()

    has_genres = "genres" in df.columns
    has_title = "title" in df.columns

    keep_cols = required_columns.copy()
    if has_title:
        keep_cols.append("title")
    if has_genres:
        keep_cols.append("genres")

    # -------------------------------
    # DATA CLEANING
    # -------------------------------
    df = df[keep_cols]

    df = df.dropna(subset=required_columns)

    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ]

    # -------------------------------
    # PARSE GENRES COLUMN
    # -------------------------------
    # genres arrives as a stringified list of dicts, e.g. "[{'id': 10749, 'name': 'Romance'}]"
    # We extract a clean list of genre names and a primary_genre for grouping/filtering.
    def extract_genre_names(raw):
        if pd.isna(raw):
            return []
        try:
            parsed = ast.literal_eval(raw)
            return [g["name"] for g in parsed if "name" in g]
        except (ValueError, SyntaxError):
            return []

    if has_genres:
        df["genre_list"] = df["genres"].apply(extract_genre_names)
        df["primary_genre"] = df["genre_list"].apply(
            lambda g: g[0] if len(g) > 0 else "Unknown"
        )

    # -------------------------------
    # CREATE TARGET COLUMN
    # -------------------------------
    df["success"] = np.where(
        df["revenue"] > df["budget"],
        1,
        0
    )

    # -------------------------------
    # SIDEBAR FILTERS
    # -------------------------------
    st.sidebar.header("Filters")

    min_rating = st.sidebar.slider(
        "Minimum Vote Average",
        0.0,
        10.0,
        0.0
    )

    df = df[df["vote_average"] >= min_rating]

    if has_genres:
        all_genres = sorted(set(g for genre_list in df["genre_list"] for g in genre_list))
        selected_genres = st.sidebar.multiselect(
            "Filter by Genre",
            options=all_genres,
            default=[]
        )
        if selected_genres:
            df = df[df["genre_list"].apply(lambda g: any(sg in g for sg in selected_genres))]

    # -------------------------------
    # KPI CARDS
    # -------------------------------
    total_movies = len(df)

    success_rate = round(
        df["success"].mean() * 100,
        2
    ) if total_movies > 0 else 0

    avg_budget = round(
        df["budget"].mean(),
        2
    ) if total_movies > 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🎬 Total Movies",
        total_movies
    )

    col2.metric(
        "✅ Success Rate",
        f"{success_rate}%"
    )

    col3.metric(
        "💰 Average Budget",
        f"${avg_budget:,.0f}"
    )

    st.divider()

    # -------------------------------
    # DATASET PREVIEW
    # -------------------------------
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    # -------------------------------
    # DESCRIPTIVE STATISTICS
    # -------------------------------
    st.subheader("Descriptive Statistics")

    st.write(df.describe())

    st.divider()

    # -------------------------------
    # SUCCESS DISTRIBUTION
    # -------------------------------
    st.subheader("Success Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    df["success"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xticklabels(["Failure", "Success"], rotation=0)
    ax.set_xlabel("Movie Status")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # BUDGET VS REVENUE
    # -------------------------------
    st.subheader("Budget vs Revenue")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df["budget"],
        df["revenue"]
    )

    ax.set_xlabel("Budget")
    ax.set_ylabel("Revenue")
    ax.set_title("Budget vs Revenue")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # GENRE TRENDS
    # -------------------------------
    if has_genres:
        st.subheader("Genre Trends")

        genre_exploded = df.explode("genre_list")
        genre_exploded = genre_exploded[genre_exploded["genre_list"].notna()]

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("**Most Common Genres**")
            genre_counts = genre_exploded["genre_list"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            genre_counts.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_xlabel("Number of Movies")
            st.pyplot(fig)

        with col_b:
            st.write("**Success Rate by Genre**")
            genre_success = genre_exploded.groupby("genre_list")["success"].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            (genre_success * 100).plot(kind="barh", ax=ax, color="green")
            ax.invert_yaxis()
            ax.set_xlabel("Success Rate (%)")
            st.pyplot(fig)

        st.divider()

    # -------------------------------
    # VOTE AVERAGE DISTRIBUTION
    # -------------------------------
    st.subheader("Vote Average Distribution")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(
        df["vote_average"],
        bins=20
    )

    ax.set_xlabel("Vote Average")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # CORRELATION HEATMAP
    # -------------------------------
    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7,6))

    heatmap = ax.imshow(
        corr,
        interpolation="nearest",
        aspect="auto"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(corr.columns)

    plt.colorbar(heatmap)

    # Annotate each cell with its correlation value
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------
    st.subheader("Statistical Analysis")

    corr_value, p_value = pearsonr(
        df["budget"],
        df["revenue"]
    )

    st.write("### Pearson Correlation (Budget vs Revenue)")

    st.write(
        f"Correlation Coefficient : {corr_value:.4f}"
    )

    st.write(
        f"P-value : {p_value:.6f}"
    )

    success_movies = df[
        df["success"] == 1
    ]["vote_average"]

    failure_movies = df[
        df["success"] == 0
    ]["vote_average"]

    t_stat, t_p = ttest_ind(
        success_movies,
        failure_movies,
        equal_var=False
    )

    st.write("### Independent T-Test (Vote Average: Success vs Failure)")
    st.write("Null Hypothesis (H0): Mean vote_average is the same for successful and unsuccessful movies.")

    st.write(
        f"T-Statistic : {t_stat:.4f}"
    )

    st.write(
        f"P-value : {t_p:.6f}"
    )

    if t_p < 0.05:
        st.write("Conclusion: Reject H0 — vote_average differs significantly between successful and unsuccessful movies.")
    else:
        st.write("Conclusion: Fail to reject H0 — no significant difference found.")

    # -------------------------------
    # CHI-SQUARE TEST (GENRE vs SUCCESS)
    # -------------------------------
    if has_genres:
        st.write("### Chi-Square Test (Primary Genre vs Success)")
        st.write("Null Hypothesis (H0): Genre and movie success are independent.")

        contingency = pd.crosstab(df["primary_genre"], df["success"])
        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency)

        st.write(f"Chi-Square Statistic : {chi2_stat:.4f}")
        st.write(f"Degrees of Freedom : {dof}")
        st.write(f"P-value : {chi2_p:.6f}")

        if chi2_p < 0.05:
            st.write("Conclusion: Reject H0 — genre and success appear to be associated.")
        else:
            st.write("Conclusion: Fail to reject H0 — no significant association found between genre and success.")

    st.divider()

    # -------------------------------
    # MACHINE LEARNING
    # -------------------------------
    st.subheader("Machine Learning")

    X = df[
        [
            "budget",
            "popularity",
            "runtime",
            "vote_average",
        ]
    ]

    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.success(
        f"Accuracy : {accuracy*100:.2f}%"
    )

    st.write("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    st.write("### Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(figsize=(4,4))

    image = ax.imshow(cm)

    plt.colorbar(image)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(
        ["Failure","Success"]
    )

    ax.set_yticklabels(
        ["Failure","Success"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center",
                fontsize=12
            )

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------
    st.write("### Feature Importance")

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # MOVIE SUCCESS PREDICTION
    # -------------------------------
    st.subheader("🎯 Predict Movie Success")

    budget_input = st.number_input(
        "Budget",
        min_value=0.0,
        value=50000000.0,
        step=1000000.0
    )

    popularity_input = st.number_input(
        "Popularity",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    runtime_input = st.number_input(
        "Runtime",
        min_value=1,
        value=120,
        step=1
    )

    vote_average_input = st.number_input(
        "Vote Average",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    if st.button("Predict"):

        input_data = pd.DataFrame(
            [[budget_input, popularity_input, runtime_input, vote_average_input]],
            columns=[
                "budget",
                "popularity",
                "runtime",
                "vote_average",
            ]
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("🎉 Prediction: This movie is likely to be Successful!")
        else:
            st.error("❌ Prediction: This movie is likely to be Unsuccessful.")

        st.write(
            f"Confidence: {max(probability) * 100:.2f}%"
        )

    st.divider()

    # -------------------------------
    # DOWNLOAD FILTERED DATA
    # -------------------------------
    st.subheader("📥 Download Filtered Dataset")

    download_df = df.drop(columns=["genre_list"], errors="ignore")

    csv = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

    st.divider()

    # -------------------------------
    # BUSINESS INSIGHTS & RECOMMENDATIONS
    # -------------------------------
    st.subheader("📊 Business Insights & Recommendations")

    top_feature = importance_df.iloc[0]["feature"]
    top_feature_score = importance_df.iloc[0]["importance"]

    budget_revenue_note = (
        "Budget and revenue move strongly together, but a bigger budget alone does not "
        "guarantee a movie clears its budget — plenty of high-budget films still land in "
        "the 'failure' zone on the scatter plot above."
        if corr_value > 0.5 else
        "Budget and revenue do not move together as strongly as expected, suggesting "
        "spending alone is a weak predictor of box-office return."
    )

    if has_genres and len(genre_success) > 0:
        best_genre = genre_success.index[0]
        best_genre_rate = genre_success.iloc[0] * 100
        genre_note = (
            f"**{best_genre}** has the highest success rate among common genres "
            f"(~{best_genre_rate:.1f}%) in the current (filtered) dataset."
        )
    else:
        genre_note = "Upload a dataset with a genres column to see genre-level recommendations."

    st.markdown(
        f"""
- **Overall success rate:** {success_rate}% of the {total_movies} movies in view turned a profit (revenue > budget).
- **Budget vs. Revenue:** Pearson correlation is {corr_value:.2f} (p={p_value:.4f}). {budget_revenue_note}
- **Strongest predictor:** The Random Forest model relies most on **{top_feature}** (importance score {top_feature_score:.2f}) when predicting success — this lines up with what the correlation heatmap and statistical tests above show.
- **Genre signal:** {genre_note}
- **Model confidence:** The current model reaches {accuracy*100:.2f}% accuracy on the held-out test set — solid as a decision-support signal, but not precise enough to be the sole basis for greenlighting a film.

**Recommendations**
1. Use budget planning alongside **{top_feature.lower()}** benchmarks rather than budget alone when forecasting a film's commercial potential.
2. Treat genre as a portfolio-balancing tool — favor genres with historically higher success rates when risk tolerance is low, and reserve high-budget bets for genres/features the model rates favorably.
3. Re-train the model periodically as new releases are added, since audience preferences and cost structures shift over time.
4. Pair this model's prediction with qualitative factors (cast, marketing spend, release timing, competition) that aren't in this dataset — it should support, not replace, human judgment.
        """
    )

    st.divider()

    # -------------------------------
    # FOOTER
    # -------------------------------
    st.markdown("---")

    st.markdown(
        """
        ### 🎬 MovieIQ Dashboard

        **Project Features**

        - ✅ Data Cleaning
        - ✅ Exploratory Data Analysis (EDA)
        - ✅ Genre Trends
        - ✅ Statistical Analysis (T-Test & Chi-Square)
        - ✅ Random Forest Classification
        - ✅ Confusion Matrix
        - ✅ Classification Report
        - ✅ Feature Importance
        - ✅ Movie Success Prediction
        - ✅ Download Filtered Dataset

        Developed using **Python**, **Pandas**, **Matplotlib**, **Scikit-learn**, **SciPy**, and **Streamlit**.
        """
    )

else:
    st.info("👆 Please upload your **movies.csv** file to begin.")