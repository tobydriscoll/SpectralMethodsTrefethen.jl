# p20.m - 2nd-order wave eq. in 2D via FFT (compare p19.m)

using PyPlot, Interpolations
# Grid and initial data:
N = 24; x = y = cheb(N)[2];
dt = 6/N^2;
xx = yy = x[end:-1:1];
plotgap = round(Int,(1/3)/dt); dt = (1/3)/plotgap;
vv = @. exp(-40*((x'-.4)^2 + y^2));
vvold = vv;  clf();

# Time-stepping by leap frog formula:
for n = 0:3*plotgap
    t = n*dt;
    if rem(n+.5,plotgap)<1     # plots at multiples of t=1/3
          i = n/plotgap+1;
          figure(i); clf();
          xxx = yyy = -1:1/16:1;
          s = interpolate((xx,yy),reduce(flipdim,vv,1:2),Gridded(Linear()));
          vvv = s[xxx,yyy];
          surf(xxx,yyy,vvv); xlim(-1,1); ylim(-1,1); zlim(-0.15,1);
          gca()[:view_init](30,-127.5);
          title("t = $t");
    end
    uxx = zeros(N+1,N+1); uyy = zeros(N+1,N+1);
    ii = 2:N;
    for i = 2:N                # 2nd derivs wrt x in each row
          v = vv[i,:]; V = [v;flipdim(v[ii],1)];
          U = real(fft(V));
          W1 = real(ifft(1im*[0:N-1;0;1-N:-1].*U)); # diff wrt theta
          W2 = real(ifft(-[0:N;1-N:-1].^2.*U));     # diff^2 wrt theta
          uxx[i,ii] = W2[ii]./(1-x[ii].^2) - x[ii].*W1[ii]./(1-x[ii].^2).^(3/2);
    end
    for j = 2:N                # 2nd derivs wrt y in each column
          v = vv[:,j]; V = [v; flipdim(v[ii],1)];
          U = real(fft(V));
          W1 = real(ifft(1im*[0:N-1;0;1-N:-1].*U));# diff wrt theta
          W2 = real(ifft(-[0:N;1-N:-1].^2.*U));    # diff^2 wrt theta
          uyy[ii,j] = W2[ii]./(1-y[ii].^2) - y[ii].*W1[ii]./(1-y[ii].^2).^(3/2);
    end
    vvnew = 2*vv - vvold + dt^2*(uxx+uyy);
    vvold = vv; vv = vvnew;
end
