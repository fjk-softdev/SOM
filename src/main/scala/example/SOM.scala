package example

import java.io.PrintWriter

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{tf, _}
import org.platanios.tensorflow.api.tensors
import org.slf4j.LoggerFactory


class SOM (val m: Int = 30, val n: Int = 30, val dim : Int, val iterations: Int = 200, val alphainit : Double = 0.3, val initRadius : Double = 1){

  // single SOM
  private val logger = Logger(LoggerFactory.getLogger("Examples for SOM"))

  logger.info(s"Building SOM model using m = $m, n = $n, number of iterations = $iterations, number of dimensions = $dim, initial learning rate = $alphainit")

  val maxIterations = tf.constant[Double](iterations).reshape(Shape(-1))
  val locations = tf.constant(node_locations(m,n))
//  val maxIterations = tf.placeholder[Double](Shape(-1),name = "maxIterations")

  val inputs = tf.placeholder[Double](Shape(-1,dim))
  val epoch = tf.placeholder[Double](Shape(),name = "epoch")


  val weights = tf.variable[Double]("weights",Shape(m*n,dim),tf.RandomNormalInitializer())

  // get bmu index
  // no sqrt used
  // example input has (1,3) and weights (400,3). -> compare all data by row. expand by 1 to 1,400,3 and 1,1,3 and get 1,400,3 after subtraction
  val diff  = tf.subtract(
    tf.expandDims(inputs,axis = 1),
    tf.expandDims(weights,axis = 0)
  )
  val bmu_index = tf.argmin(
    tf.sum(
      tf.pow(diff,2.0)
      ,axes = 2,keepDims = false
    ),1,INT32
  )


  // get bmu location
  val bmu_loc = tf.gather(locations,bmu_index,axis = 0)

  val radius = tf.subtract(
    initRadius,tf.multiply(initRadius,tf.divide(epoch,tf.add(maxIterations,1.0)))
  )

  val alpha = tf.multiply(alphainit,
    tf.subtract(1.0,
      tf.divide(epoch,maxIterations)
    )
  )

  val distdiffonNodes = tf.subtract(tf.expandDims(bmu_loc,axis = 1),tf.expandDims(locations,axis = 0))

  val bmu_distance_squares = tf.sqrt(
    tf.sum(
      tf.pow(
        distdiffonNodes,2.0
      ),keepDims = true,axes = 2
    )
  )

  val neighborhood_func = tf.exp(
    tf.divide(
      tf.multiply(bmu_distance_squares,-1.0),
      radius)
  )

  val learning_rate_op =
    tf.multiply(
      alpha,neighborhood_func
    )

  val weightvector = tf.sum(tf.multiply(diff,learning_rate_op),axes = 0,keepDims = false)

  val newweights = tf.add(weights,weightvector)

  val training_op = weights.assign(newweights)

  //get the session and get init stuff started
  val session = Session()
  session.run(targets = tf.globalVariablesInitializer())

  def training() = {
    logger.info("Training stuff")
    val trainBatch = getbatch()
    val nodelocs  = node_locations(m,n)
    val newbatch =  Tensor[Double](trainBatch).reshape(Shape(400,2))

    for (i <- 1 to iterations) {
      val feed1 = FeedMap(Map(inputs -> newbatch, epoch -> Tensor[Double](i)))
      val trainLoss = session.run(feeds = feed1, fetches = weights.value,targets = training_op)
    }
    val test = session.run(fetches = weights.value)

    logger.info("finished training of nodes")
    createMap(test,Tensor[Double](2.0))
  }

  def getbatch() : Array[Array[Double]] = {
    //manual input
    val inputs = Array.ofDim[Double](400,2)

    val r = scala.util.Random
    for(i <- 0 to 100){
      inputs(i)(0) = 0.0
      inputs(i)(1) = r.nextDouble()
    }
    for(i <- 101 to 399){
      inputs(i)(0) = r.nextDouble()
      inputs(i)(1) = 0.0
    }
/*
          inputs(0)(0) = 1
          inputs(0)(1) = 1
          inputs(0)(2) = 1

          inputs(1)(0) = 1
          inputs(1)(1) = 1
          inputs(1)(2) = 0

          inputs(2)(0) = 0
          inputs(2)(1) = 1
          inputs(2)(2) = 0

          inputs(3)(0) = 1
          inputs(3)(1) = 0
          inputs(3)(2) = 0
      */
    inputs
  }

  def node_locations(m: Int, n: Int) : Tensor[Double] = {
    // matrix representation of nodes
    val buffer = (for(i <- 1 to m; j <- 1 to n ) yield Tensor[Double](i,j))
    buffer
  }

  def createMap(input: Tensor[Double],qradius: Tensor[Double] = 2) : Unit = {

    logger.info("creating map from trained nodes")
    val rsdiff = tf.sum(tf.pow(tf.subtract(tf.expandDims(input,axis = 0),tf.expandDims(input,axis=1)),2.0),axes = 2,keepDims = false)
    val nsdiff = (tf.lessEqual(tf.sum(tf.pow(tf.subtract(tf.expandDims(locations,axis = 0),tf.expandDims(locations,axis=1)),2.0),axes = 2,keepDims = false),1.0)).castTo(Double)
    val result = tf.mean(tf.multiply(rsdiff,nsdiff),axes = 1)

    val evresult = (session.run(fetches = result)).entriesIterator.toArray


    val test = (for(i <- 1 to m;j<-1 to n) yield ( (i,j),evresult((i-1)*n+j-1) ))
    val pw = new PrintWriter("SOM_Map")

      test.foreach{
        case(nodelocs,distance) => {
          pw.write(nodelocs._1.toString + "\t" + nodelocs._2.toString + "\t"+ distance.toString +"\n")
        }
      }
    pw.close
    logger.info("map has been created. Check filename SOM_Map")
  }

}
