#include "ga/algorithm.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

static auto f1(double x) -> double
{
  return -1.0 /(fabs(x) + 0.001);
}


//Rastrigins function, for function optimization.
//
//find the global minima (0,0) of the rastrigins function:
//ras =  (20 + x(1)^2 + x(2)^2 - 10*(cos(2*pi*x(1)) +    cos(2*pi*x(2)) ) );
static auto ras(double x1, double x2) -> double { return (20.0f + x1*x1 + x2*x2 - 10.0*(cos(2.0*M_PI*x1) + cos(2*M_PI*x2)) ); }

template <typename G> static auto drand(G& g) -> double
{
  return std::generate_canonical<double, std::numeric_limits<double>::digits>(g);
}

class individual
{
public:  
  individual(const double _x1, const double _x2)
    : x1{_x1},x2{_x2}
  {
  }
  individual(const std::pair<double,double>& p) : individual(p.first,p.second) {}
  

  friend auto operator<<(std::ostream& os, const individual& x) -> std::ostream&
  {
    return os << "x = " << x.x1  << ", "<< x.x2 <<"\tf(x) = [" << f1(ras(x.x1,x.x2)) << ']';
  }

  double x1, x2;
};

class problem
{
public:
  using individual_type = individual;
  using generator_type = std::mt19937;
  using fitness_type = double;

  auto evaluate(const individual_type& x, generator_type&) const -> double
  {
    return f1(ras(x.x1,x.x2));
  }

  auto mutate(individual_type& x, generator_type& g) const -> void
  {
    if (drand(g) < 0.1) {
      x.x1 += drand(g)*(2.0*(drand(g)>0.5)-1.0);
      x.x2 += drand(g)*(2.0*(drand(g)>0.5) - 1.0);
    } else {
      x.x1 -= drand(g)*(2.0*(drand(g)>0.5)-1.0);
      x.x2 -= drand(g)*(2.0*(drand(g)>0.5) - 1.0);      
    }
  }

  auto recombine(const individual_type& a, const individual_type& b,
                 generator_type& g) const -> std::array<individual_type, 2u>
  {
    if (drand(g) < 0.4)
      return {{b, a}};
    return {{a, b}};
  };
};

auto operator<<(std::ostream& os, const ga::algorithm<problem>::solution_type& s)
  -> std::ostream&
{
  return os << s.x << ",\tfitness = " << s.fitness;
}

int main()
{
  auto initial_population = std::vector<individual>{};
  auto generator = std::mt19937{17u};

  initial_population.reserve(10u);
  std::generate_n(std::back_inserter(initial_population), initial_population.capacity(),
                  [&] { return std::make_pair(2*drand(generator),2*drand(generator)); });

  auto model = ga::make_algorithm(problem{}, std::move(initial_population), 5u,
                                  std::move(generator));

  std::cout << std::setprecision(6) << std::fixed;
  for (auto t = 0u; t <= 10; ++t)
  {
    std::cout << "=== Iteration " << t << " ===" << std::endl;
    std::cout << "Population: " << std::endl;
    for (const auto& ind : model.population())
      std::cout << '\t' << ind << std::endl;
    model.iterate();
  }
}
