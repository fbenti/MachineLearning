import source.Data as Da
import source.pcanalysis as pca
import source.regression as re
import source.classification as cl


if __name__ == '__main__':

    # fetch data through data object
    data = Da.Data()

    


    # cl.logistic_regression(data, 10)
    # cl.k_nearest_neighbours(data, 10, 40)
    # cl.baseline(data, 5)
    # cl.two_layer_cross_validation(data, 10, 5, True)
    # cl.mcnemera(data, 10, True)
    # cl.train_log_model(data, 22.229964825261945, True)

    cl.test(data)
