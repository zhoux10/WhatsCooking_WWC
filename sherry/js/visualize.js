$(function () {
/*
trainData contains an array of Object:
{
  id: recipe's id (#)
  cuisine: type of cuisine (string)
  ingredients: list of ingredients (array of strings)
}
*/

// Process data by creating an array of ingredients that d3 can use to create bars (ingredientsToGraph):
// Also create an array of cuisines and the number of ingredients in them (numberOfIngredientsToGraph)
var ingredientsToIdx = Object.create(null),
    cuisineToIdx = Object.create(null),
    ingredientsToGraph = [], numberOfIngredientsToGraph = [],
    ingredientsArray, cuisine;
trainData.forEach(function (recipe) {
  cuisine = recipe.cuisine;
  ingredientsArray = recipe.ingredients;

  // Add number of ingredients to an array of objects
  createOrAdd(  {
      idxObject: cuisineToIdx,
      array: numberOfIngredientsToGraph,
      objectKey: cuisine,
      newObject: { name: cuisine, numberIngredients: [ingredientsArray.length]},
      addCb: function (object) {
        object.numberIngredients.push(ingredientsArray.length);
      }
    }); /*createOrAdd*/
  ingredientsArray.forEach(function (ingredient) {
    createOrAdd(  {
        idxObject: ingredientsToIdx,
        array: ingredientsToGraph,
        objectKey: ingredient,
        newObject: { name: ingredient,
                     numberCuisines: { cuisine: 1 } },
        addCb: function (object) {
          var cuisineInIngredients = object.numberCuisines[cuisine];
          if (cuisineInIngredients) {
            cuisineInIngredients += 1;
          } else {
            object.numberCuisines[cuisine] = 1;
          }
        }
      }); /*createOrAdd*/
  }); /*ingredientsArray*/

  /* Expects hash of:
  {
    idxObject: hash of object name to idx in array
    array: array to be added to
    objectKey: string
    newObject: object or array
    addCb: function that accepts object
  }
  */
  function createOrAdd (c) {
    var idx = c.idxObject[c.objectKey],
        object;

    if (!idx) {
      c.idxObject[c.objectKey] = c.array.length;
      c.array.push(c.newObject);
    } else {
      object = c.array[idx];
      c.addCb(object);
    }
  }
}); /*trainData*/

createBoxPlot(".container");

function createBoxPlot (div) {
  var labels = true; // show the text labels beside individual boxplots?

  var margin = {top: 30, right: 50, bottom: 80, left: 50};
  var  width = 1500 - margin.left - margin.right;
  var height = 400 - margin.top - margin.bottom;

  var min = Infinity,
      max = -Infinity;

  // parse in the data
  	var data = numberOfIngredientsToGraph.map(function (cuisine) {
  		var rowMax = Math.max.apply(null, cuisine.numberIngredients);
  		var rowMin = Math.min.apply(null, cuisine.numberIngredients);

      if (rowMax > max) max = rowMax;
      if (rowMin < min) min = rowMin;

      return [cuisine.name, cuisine.numberIngredients];
    });

  	var chart = d3.box()
  		.whiskers(iqr(1.5))
  		.height(height)
  		.domain([min, max])
  		.showLabels(labels);

  	var svg = d3.select(div).append("svg")
  		.attr("width", width + margin.left + margin.right)
  		.attr("height", height + margin.top + margin.bottom)
  		.attr("class", "box")
  		.append("g")
  		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  	// the x-axis
  	var x = d3.scale.ordinal()
                		.domain( data.map(function(d) {
                                        console.log(d); return d[0];
                                      } ) )
                		.rangeRoundBands([0 , width], 0.7, 0.3);

  	var xAxis = d3.svg.axis()
  		.scale(x)
  		.orient("bottom");

  	// the y-axis
  	var y = d3.scale.linear()
  		.domain([min, max])
  		.range([height + margin.top, 0 + margin.top]);

  	var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left");

  	// draw the boxplots
  	svg.selectAll(".box")
        .data(data)
  	  .enter().append("g")
  		.attr("transform", function(d) { return "translate(" +  x(d[0])  + "," + margin.top + ")"; } )
        .call(chart.width(x.rangeBand()));


  	// add a title
  	svg.append("text")
          .attr("x", (width / 2))
          .attr("y", 0 + (margin.top / 2))
          .attr("text-anchor", "middle")
          .style("font-size", "18px")
          //.style("text-decoration", "underline")
          .text("Number of Ingredients/Cuisine");

  	 // draw y axis
  	svg.append("g")
          .attr("class", "y axis")
          .call(yAxis)
  		.append("text") // and text1
  		  .attr("transform", "rotate(-90)")
  		  .attr("y", 6)
  		  .attr("dy", ".71em")
  		  .style("text-anchor", "end")
  		  .style("font-size", "16px")
  		  .text("Number of Ingredients");

  	// draw x axis
  	svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (height  + margin.top + 10) + ")")
        .call(xAxis)
  	  .append("text")             // text label for the x axis
          .attr("x", (width / 2) )
          .attr("y",  20 )
  		.attr("dy", ".71em")
          .style("text-anchor", "middle")
  		.style("font-size", "16px")
          .text("Ingredients");
  }

  // Returns a function to compute the interquartile range.
  function iqr(k) {
    return function(d, i) {
      var q1 = d.quartiles[0],
          q3 = d.quartiles[2],
          iqr = (q3 - q1) * k,
          i = -1,
          j = d.length;
      while (d[++i] < q1 - iqr);
      while (d[--j] > q3 + iqr);
      return [i, j];
    };
  }

});
