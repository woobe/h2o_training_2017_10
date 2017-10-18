# Classification Example based on Smartphone Sensors Data


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


# or import the files locally if you have them
train = h2o.importFile("./examples/clustering/water_treatment_plant.csv")


# ------------------------------------------------------------------------------
# Have a quick look
# ------------------------------------------------------------------------------

h2o.describe(train)
h2o.summary(train)



# ------------------------------------------------------------------------------
# Define features for clustering
# ------------------------------------------------------------------------------

# Remove "name"
features = setdiff(colnames(train), "name")


# ------------------------------------------------------------------------------
# Build K-Means Model
# ------------------------------------------------------------------------------

model_km = h2o.kmeans(x = features,
                      training_frame = train,
                      k = 10,
                      estimate_k = TRUE,
                      seed = 1234)
print(model_km)

clusters = h2o.predict(model_km, newdata = train)

new_data = h2o.cbind(clusters, train)
df_new_data = as.data.frame(new_data)

