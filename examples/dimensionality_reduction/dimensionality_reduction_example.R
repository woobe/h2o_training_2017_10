# Example based on Smartphone Sensors Data (again)


# ------------------------------------------------------------------------------
# Start H2O
# ------------------------------------------------------------------------------

library(h2o)
h2o.init(nthreads = -1)

# ------------------------------------------------------------------------------
# Import file from internet
# ------------------------------------------------------------------------------

# Import from GitHub
# train = h2o.importFile("https://github.com/woobe/h2o_training_2017_10/blob/master/examples/classification/smartphone_sensors_train.csv.zip?raw=true")
# test = h2o.importFile("https://github.com/woobe/h2o_training_2017_10/blob/master/examples/classification/smartphone_sensors_test.csv.zip?raw=true")

# or import the files locally if you have them
train = h2o.importFile("./examples/classification/smartphone_sensors_train.csv.zip")
test = h2o.importFile("./examples/classification/smartphone_sensors_test.csv.zip")


# ------------------------------------------------------------------------------
# Define features and target
# ------------------------------------------------------------------------------

features = setdiff(colnames(train), "activity")
target = "activity"


# ------------------------------------------------------------------------------
# Build a PCA Model
# ------------------------------------------------------------------------------

# H2O PCA
model_pca = h2o.prcomp(x = features,
                       training_frame = train,
                       k = 5,     # your choice
                       impute_missing = TRUE,
                       seed = 1234)


# Create new features
new_features_train = h2o.predict(model_pca, train)
head(new_features_train)

new_features_test = h2o.predict(model_pca, test)
head(new_features_test)
