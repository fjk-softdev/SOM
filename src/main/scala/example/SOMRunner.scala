package example

import java.io.PrintWriter

import org.platanios.tensorflow.api.{INT32, Tensor, tf}
import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.{tf, _}
import org.slf4j.LoggerFactory

object SOMRunner extends App {

  val singlesom = new SOM(dim = 2,m = 20,n=20,initRadius = 1,alphainit = 0.5,iterations = 1000)
  singlesom.training()


}