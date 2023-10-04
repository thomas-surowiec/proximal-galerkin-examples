// 
//  MFEM implementation of the proximal Galerkin method for advection-diffusion introduced in [1].
//
//    [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//       preserving finite element method for pointwise bound constraints.
//       arXiv:2307.12444 [math.NA]
//
//  To build, install MFEM and place this file in the `/mfem/examples` directory. Then compile with 
//  the command `make advection_diffusion`.
//
//  Note: This file requires MFEM to be built with UMFPACK
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double eps = 1e-2;
double Ramp_BC(const Vector &pt);
double EJ_exact_solution(const Vector &pt);

double lnit(double x)
{
   double tol = 1e-13;
   x = min(max(tol,x),1.0-tol);
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double min_val_=-1e10, double max_val_=1e10)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExpitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   ExpitGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dExpitdxGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // This is a serial implementation (No MPI)

   // 1. Parse command-line options.
   int order = 1;
   int ref_levels = 3;
   int max_it = 100;
   double tol = 1e-4;
   double alpha = 0.01;
   double rho = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Bregman step size alpha");
   args.AddOption(&rho, "-rho", "--rho",
                  "Proximal step size rho");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   const char *mesh_file = "../data/inline-quad.mesh";
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   H1_FECollection L2fec(order, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec);

   cout << "Number of H1 finite element unknowns: "
        << H1fes.GetTrueVSize() 
        << "\nNumber of L2 finite element unknowns: "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   GridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x,offsets[0]);
   delta_psi_gf.MakeRef(&L2fes,x,offsets[1]);
   delta_psi_gf = 0.0;

   // 5. Determine the list of true (i.e., conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_tdof_list_L2;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (H1fes.GetTrueVSize() == L2fes.GetTrueVSize())
      {
         L2fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_L2);
      }
   }

   // 6. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_old_gf(&H1fes);
   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;

   // 7. Define various coefficients, including the function coefficient
   //    for the solution
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   Vector beta(dim);
   beta(0) = 1.0;
   beta(1) = 0.0;
   ConstantCoefficient f(0.0);
   VectorConstantCoefficient beta_coeff(beta);
   ConstantCoefficient eps_coeff(eps);

   FunctionCoefficient bdry_coef(EJ_exact_solution);
   u_gf.ProjectCoefficient(bdry_coef);

   // 8. Solve linear system to get initial guess
   LinearForm b(&H1fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   BilinearForm a(&H1fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(eps_coeff));
   a.AddDomainIntegrator(new ConvectionIntegrator(beta_coeff));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X, B);

   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);

   a.RecoverFEMSolution(X, b, u_gf);
   u_old_gf = u_gf;

   LnitGridFunctionCoefficient lnit_u(u_gf);
   psi_gf.ProjectCoefficient(lnit_u);
   psi_old_gf = psi_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   socketstream sol_sock2(vishost, visport);
   sol_sock2.precision(8);

   // 9. Initialize the slack variable ψₕ = lnit(uₕ)
   ExpitGridFunctionCoefficient expit_psi(psi_gf);
   GridFunction u_alt_gf(&L2fes);
   u_alt_gf.ProjectCoefficient(expit_psi);
   sol_sock << "solution\n" << mesh << u_gf <<
               "keys a\n" <<
               "window_title 'Discrete solution u_h'" << flush;

   // 10. Save data
   mfem::ParaViewDataCollection paraview_dc("Advection_diffusion", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("u",&u_gf);
   paraview_dc.RegisterField("tilde_u",&u_alt_gf);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.Save();

   // 11. Iterate
   int k;
   int total_iterations = 0;
   double increment_u = 1e-3;
   for (k = 0; k < max_it; k++)
   {
      GridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 10; j++)
      {
         total_iterations++;

         LinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ExpitGridFunctionCoefficient expit_psi(psi_gf);
         dExpitdxGridFunctionCoefficient dexpitdx_psi(psi_gf);
         ProductCoefficient neg_dexpitdx_psi(-1.0, dexpitdx_psi);

         GridFunctionCoefficient u_old_cf(&u_old_gf);
         GradientGridFunctionCoefficient grad_u_old(&u_old_gf);
         InnerProductCoefficient beta_grad_u_old(beta_coeff, grad_u_old);
         ScalarVectorProductCoefficient alpha_pde_residual1(alpha*(1.0/rho - eps), grad_u_old);
         SumCoefficient alpha_pde_residual2(beta_grad_u_old, f, -alpha, alpha);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

         b0.AddDomainIntegrator(new DomainLFGradIntegrator(alpha_pde_residual1));
         b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_pde_residual2));
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.Assemble();

         b1.AddDomainIntegrator(new DomainLFIntegrator(expit_psi));
         b1.Assemble();

         BilinearForm a00(&H1fes);
         ConstantCoefficient alpha_rho_cf(alpha/rho);
         a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_rho_cf));
         a00.Assemble();
         a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         MixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         a10.EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
         a10.Finalize();
         SparseMatrix &A10 = a10.SpMat();

         SparseMatrix &A01 = *Transpose(A10);

         BilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(neg_dexpitdx_psi));
         a11.Assemble();
         a11.EliminateEssentialBC(ess_bdr,x.GetBlock(1),rhs.GetBlock(1),mfem::Operator::DIAG_ONE);
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat(); 

         BlockMatrix A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         SparseMatrix * A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs,x);

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         u_alt_gf.ProjectCoefficient(expit_psi);
         sol_sock2 << "solution\n" << mesh << u_alt_gf <<
                      "keys a\n" <<
                      "window_title 'Discrete solution \\tilde{u}_h'" << flush;


         double factor = 0.5;
         delta_psi_gf *= factor;
         psi_gf += delta_psi_gf;

         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_u/100.0)
         {
            break;
         }

      }
      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;

      delta_psi_gf = psi_gf;
      delta_psi_gf -= psi_old_gf;
      delta_psi_gf = 0.0;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;
   
      paraview_dc.SetCycle(k);
      paraview_dc.SetTime((double)k);
      paraview_dc.Save();

      if (increment_u < tol)
      {
         break;
      }
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << endl;

   return 0;
}

double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val)));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, expit(val)));
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, dexpitdx(val)));
}

double Ramp_BC(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double tol = 1e-10;
   double eps = 0.05;

   if (  (abs(y) < tol && x >= 0.2)
      || (abs(x-1.0) < tol)
      || (abs(y-1.0) < tol) )
   {
      return 0.0;
   }
   else if (  (abs(x) < tol && y <= 1.0 - eps)
           || (abs(y) < tol && x <= 0.2 - eps) )
   {
      return 1.0;
   }
   else if (x >= (0.2 - eps) && abs(y) < tol)
   {
      return (0.2 - x)/eps;
   }
   else if (y >= (1.0 - eps) && abs(x) < tol)
   {
      return  (1.0 - y)/eps;
   }
   else
   {
      return 0.5;
   }
}

double EJ_exact_solution(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double lambda = M_PI*M_PI*eps;
   double r1 = (1.0 + sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);
   double r2 = (1.0 - sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);

   double num = exp(r2 * (x - 1.0)) - exp(r1 * (x-1.0));
   double denom = exp(-r2) - exp(-r1);
   // double denom = r1 * exp(-r2) - r2 * exp(-r1);

   double scale = 0.5;
   // double scale = (r1 * exp(-r2) - r2 * exp(-r1)) / (exp(-r2) - exp(-r1));

   return scale * num / denom * cos(M_PI * y) + 0.5;
   
}