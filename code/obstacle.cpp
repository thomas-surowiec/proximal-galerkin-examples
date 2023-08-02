//                  Based on MFEM Example 36 - Parallel Version
//
//
// Add file to mfem/examples and compile with: make obstacle
//
// Sample runs: mpirun -np 2 obstacle -o 2
//              mpirun -np 2 obstacle -o 2 -r 4
//              mpirun -np 4 obstacle -o 2 -tol 1e-2 -mi 100 -r 4
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ 0 in H¹ .
//
//              After solving to a specified tolerance, the numerical
//              solution is compared to a closed-form exact solution to
//              assess accuracy.
//
//              The problem is discretized and solved using the proximal
//              Galerkin finite element method, introduced by Keith and
//              Surowiec [1].
//
//
// [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double null_obstacle(const Vector &pt);
double exact_solution_poly(const Vector &pt);
void exact_solution_gradient_poly(const Vector &pt, Vector &grad);
double rhs_poly(const Vector &pt);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                    double min_val_=-1e8)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                      double min_val_=0.0, double max_val_=1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int order = 1;
   int max_it = 10;
   double tol = 1e-3;
   int ref_levels = 3;
   double alpha0 = 1.0;
   double r = 1.1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  "isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha0, "-step", "--step",
                  "Initial step size alpha");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file.
   const char *mesh_file = "../data/disc-nurbs.mesh";
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Postprocess the mesh.
   // 3A. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   // 3C. Rescale the domain to a unit circle (radius = 1).
   GridFunction *nodes = mesh.GetNodes();
   double scale = 2*sqrt(2);
   *nodes /= scale;

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection H1fec(order+1, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   int num_dofs_H1 = H1fes.GetTrueVSize();
   MPI_Allreduce(MPI_IN_PLACE, &num_dofs_H1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int num_dofs_L2 = L2fes.GetTrueVSize();
   MPI_Allreduce(MPI_IN_PLACE, &num_dofs_L2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      cout << "Number of H1 finite element unknowns: "
           << num_dofs_H1 << endl;
      cout << "Number of L2 finite element unknowns: "
           << num_dofs_L2 << endl;
   }

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   Array<int> toffsets(3);
   toffsets[0] = 0;
   toffsets[1] = H1fes.GetTrueVSize();
   toffsets[2] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   BlockVector tx(toffsets), trhs(toffsets);
   tx = 0.0; trhs = 0.0;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> empty;
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Define an initial guess for the solution.
   auto IC_func = [](const Vector &x)
   {
      double r0 = 1.0;
      double rr = 0.0;
      for (int i=0; i<x.Size(); i++)
      {
         rr += x(i)*x(i);
      }
      if (x(0) > 0)
      {
         return r0*r0 - rr + pow(x(0),4);
      }
      else
      {
         return r0*r0 - rr;
      }
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   ParGridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x.GetBlock(0).GetData());
   delta_psi_gf.MakeRef(&L2fes,x.GetBlock(1).GetData());
   delta_psi_gf = 0.0;

   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   FunctionCoefficient exact_coef(exact_solution_poly);
   VectorFunctionCoefficient exact_grad_coef(dim,exact_solution_gradient_poly);
   FunctionCoefficient IC_coef(IC_func);
   FunctionCoefficient f(rhs_poly);
   FunctionCoefficient obstacle(null_obstacle);
   u_gf.ProjectCoefficient(IC_coef);
   u_old_gf = u_gf;

   // 9. Initialize the slack variable ψₕ = exp(uₕ)
   LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
   psi_gf.ProjectCoefficient(ln_u);
   psi_old_gf = psi_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 10. Iterate
   int k;
   int total_iterations = 0;
   double increment_u = 0.1;
   double alpha = alpha0;
   double H1_error_old = 0.0;
   for (k = 0; k < max_it; k++)
   {
      alpha = alpha0*pow(r,k);

      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      if (myid == 0)
      {
         mfem::out << "\nOUTER ITERATION " << k+1 << endl;
      }

      int j;
      for ( j = 0; j < 10; j++)
      {
         total_iterations++;

         ConstantCoefficient alpha_cf(alpha);

         ParLinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
         ProductCoefficient neg_exp_psi(-1.0,exp_psi);
         GradientGridFunctionCoefficient grad_u_old(&u_old_gf);
         ProductCoefficient alpha_f(alpha, f);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

         b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.Assemble();

         b1.AddDomainIntegrator(new DomainLFIntegrator(exp_psi));
         b1.AddDomainIntegrator(new DomainLFIntegrator(obstacle));
         b1.Assemble();

         ParBilinearForm a00(&H1fes);
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_cf));
         a00.Assemble();
         HypreParMatrix A00;
         a00.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0),
                              A00, tx.GetBlock(0), trhs.GetBlock(0));


         ParMixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         HypreParMatrix A10;
         a10.FormRectangularLinearSystem(ess_tdof_list, empty, x.GetBlock(0),
                                         rhs.GetBlock(1),
                                         A10, tx.GetBlock(0), trhs.GetBlock(1));

         HypreParMatrix *A01 = A10.Transpose();

         ParBilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
         ConstantCoefficient eps_cf(-1e-6);
         if (order == 1)
         {
            a11.AddDomainIntegrator(new MassIntegrator(eps_cf));
         }
         else
         {
            a11.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
         }
         a11.Assemble();
         a11.Finalize();
         HypreParMatrix A11;
         a11.FormSystemMatrix(empty, A11);

         BlockOperator A(toffsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,A01);
         A.SetBlock(1,1,&A11);
         
         /// UNCOMMENT THE FOLLOWING TO USE A DIRECT SOLVER (best results)
         // // DIRECT solver
         // Array2D<HypreParMatrix *> BlockA(2,2);
         // BlockA(0,0) = &A00;
         // BlockA(0,1) = A01;
         // BlockA(1,0) = &A10;
         // BlockA(1,1) = &A11;
         // HypreParMatrix * Ah = HypreParMatrixFromBlocks(BlockA);

         // MUMPSSolver mumps;
         // mumps.SetPrintLevel(0);
         // mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         // mumps.SetOperator(*Ah);
         // mumps.Mult(trhs,tx);
         // delete Ah;

         ///// COMMENT THE FOLLOWING WHEN USING A DIRECT SOLVER
         // Iterative solver
         BlockDiagonalPreconditioner prec(toffsets);
         HypreBoomerAMG P00(A00);
         P00.SetPrintLevel(0);
         HypreSmoother P11(A11);
         prec.SetDiagonalBlock(0,&P00);
         prec.SetDiagonalBlock(1,&P11);

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetPrintLevel(-1);
         gmres.SetRelTol(1e-8);
         gmres.SetMaxIter(20000);
         gmres.SetKDim(500);
         gmres.SetOperator(A);
         gmres.SetPreconditioner(prec);
         gmres.Mult(trhs,tx);

         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         delta_psi_gf.SetFromTrueDofs(tx.GetBlock(1));

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         if (visualization)
         {
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
         }

         if (myid == 0)
         {
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         delete A01;

         if (Newton_update_size < increment_u)
         {
            break;
         }
      }

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      if (myid == 0)
      {
         mfem::out << "Number of Newton iterations = " << j+1 << endl;
         mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;
      }

      double H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);

      double tmp = abs(H1_error - H1_error_old)/(H1_error + 1e-12);
      if (tmp < tol || k == max_it-1)
      {
         break;
      }
      H1_error_old = H1_error;
      
      if (myid == 0)
      {
         mfem::out << "H1-error  (|| u - uₕᵏ||)       = " << H1_error << endl;
      }

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

   }

   if (myid == 0)
   {
      mfem::out << "\n Outer iterations: " << k+1
                << "\n Total iterations: " << total_iterations
                << "\n dofs:             " << num_dofs_H1 + num_dofs_L2
                << endl;
   }

   // 11. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);

      ParGridFunction error_gf(&H1fes);
      error_gf.ProjectCoefficient(exact_coef);
      error_gf -= u_gf;

      err_sock << "parallel " << num_procs << " " << myid << "\n";
      err_sock << "solution\n" << pmesh << error_gf << "window_title 'Error'"  <<
               flush;
   }

   ParGridFunction u_alt_gf(&L2fes);
   {
      double L2_error = u_gf.ComputeL2Error(exact_coef);
      double H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);

      ExponentialGridFunctionCoefficient u_alt_cf(psi_gf,obstacle);
      u_alt_gf.ProjectCoefficient(u_alt_cf);
      double L2_error_alt = u_alt_gf.ComputeL2Error(exact_coef);

      psi_old_gf -= psi_gf;
      psi_old_gf /= alpha;
      double lambda_error = psi_old_gf.ComputeL2Error(zero);

      if (myid == 0)
      {
         mfem::out << "\n Final L2-error (|| u - uₕ||)          = " << L2_error <<
                   endl;
         mfem::out << " Final H1-error (|| u - uₕ||)          = " << H1_error << endl;
         mfem::out << " Final L2-error (|| u - ϕ - exp(ψₕ)||) = " << L2_error_alt <<
                   endl;
         mfem::out << " Final L2-error (|| λ - λₕ ||) = " << lambda_error << endl;
      }
   }

   // END: Save data in the ParaView format
   ParaViewDataCollection paraview_dc("ex34p", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("u",&u_gf);
   paraview_dc.RegisterField("u_tilde",&u_alt_gf);
   paraview_dc.RegisterField("lambda",&psi_old_gf);
   paraview_dc.Save();

   return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, exp(val) + obstacle->Eval(T, ip)));
}

double null_obstacle(const Vector &pt)
{
   return 0.0;
}

double exact_solution_poly(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return pow(x,4);
   }
   else
   {
      return 0.0;
   }
}

void exact_solution_gradient_poly(const Vector &pt, Vector &grad)
{
   double x = pt(0);

   grad(1) = 0.0;
   if (x > 0.0)
   {
      grad(0) =  4.0 * pow(x,3);
   }
   else
   {
      grad(0) = 0.0;
   }
}

double rhs_poly(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return -12.0 * pow(x,2);
   }
   else
   {
      return 0.0;
   }
}