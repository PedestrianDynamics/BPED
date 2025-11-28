import numpy as np
from scipy import special, integrate
import matplotlib.pyplot as plt


def gamma_cdf(x, a, b):
    """
    Gamma distribution CDF: F(x|a, b) = (1 / (b^a * Γ(a))) * ∫[0 to x] t^(a-1) * exp(-t/b) dt
    
    This uses the regularized incomplete gamma function P(a, x/b).
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the CDF
    a : float
        Shape parameter (a > 0)
    b : float
        Scale parameter (b > 0)
    
    Returns:
    --------
    float or array
        CDF value(s)
    """
    x = np.asarray(x)
    # gammainc(a, z) computes P(a,z) = γ(a,z)/Γ(a) where γ(a,z) = ∫[0 to z] t^(a-1)*e^(-t) dt
    # For our parameterization with scale b, we need P(a, x/b)
    if np.any(x < 0):
        return np.where(x < 0, 0, special.gammainc(a, x / b))
    return special.gammainc(a, x / b)


def gamma_pdf(x, a, b):
    """
    Gamma distribution PDF: f(x|a, b) = (1 / (b^a * Γ(a))) * x^(a-1) * exp(-x/b)
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the PDF
    a : float
        Shape parameter (a > 0)
    b : float
        Scale parameter (b > 0)
    
    Returns:
    --------
    float or array
        PDF value(s)
    """
    x = np.asarray(x)
    if np.any(x < 0):
        return np.where(x < 0, 0, (1 / (b**a * special.gamma(a))) * x**(a-1) * np.exp(-x/b))
    return (1 / (b**a * special.gamma(a))) * x**(a-1) * np.exp(-x/b)


def lognormal_cdf(x, a, b):
    """
    Lognormal distribution CDF: F(x|a, b) = (1 / (b*√(2π))) * ∫[0 to x] exp(-(ln(t) - a)² / (2b²)) / t dt
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the CDF
    a : float
        Location parameter (mean of log(x))
    b : float
        Scale parameter (standard deviation of log(x), b > 0)
    
    Returns:
    --------
    float or array
        CDF value(s)
    """
    if np.any(x <= 0):
        return np.where(x <= 0, 0, 0.5 * (1 + special.erf((np.log(x) - a) / (b * np.sqrt(2)))))
    return 0.5 * (1 + special.erf((np.log(x) - a) / (b * np.sqrt(2))))


def lognormal_pdf(x, a, b):
    """
    Lognormal distribution PDF: f(x|a, b) = (1 / (x*b*√(2π))) * exp(-(ln(x) - a)² / (2b²))
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the PDF
    a : float
        Location parameter (mean of log(x))
    b : float
        Scale parameter (standard deviation of log(x), b > 0)
    
    Returns:
    --------
    float or array
        PDF value(s)
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        return np.where(x <= 0, 0, (1 / (x * b * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - a)**2 / (2 * b**2)))
    return (1 / (x * b * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - a)**2 / (2 * b**2))


def loglogistic_cdf(x, a, b):
    """
    Loglogistic distribution CDF (Logistic distribution on ln(time)):
    ln(X) ~ Logistic(μ=a, s=b)
    F(x) = 1 / (1 + exp(-(ln(x) - a) / b))
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the CDF
    a : float
        Location parameter on log scale (μ)
    b : float
        Scale parameter on log scale (s)
    
    Returns:
    --------
    float or array
        CDF value(s)
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        return np.where(x <= 0, 0, 1 / (1 + np.exp(-(np.log(x) - a) / b)))
    return 1 / (1 + np.exp(-(np.log(x) - a) / b))


def loglogistic_pdf(x, a, b):
    """
    Loglogistic distribution PDF (Logistic distribution on ln(time)):
    f(x) = (1/(b*x)) * exp((ln(x)-a)/b) / (1 + exp((ln(x)-a)/b))²
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the PDF
    a : float
        Location parameter on log scale (μ)
    b : float
        Scale parameter on log scale (s)
    
    Returns:
    --------
    float or array
        PDF value(s)
    """
    x = np.asarray(x)
    z = (np.log(x) - a) / b
    exp_z = np.exp(z)
    if np.any(x <= 0):
        return np.where(x <= 0, 0, (1 / (b * x)) * exp_z / (1 + exp_z)**2)
    return (1 / (b * x)) * exp_z / (1 + exp_z)**2


def weibull_cdf(x, a, b):
    """
    Weibull distribution CDF: F(x|a, b) = ∫[0 to x] ba^(-b) * t^(b-1) * exp(-(t/a)^b) dt
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the CDF
    a : float
        Scale parameter (a > 0)
    b : float
        Shape parameter (b > 0)
    
    Returns:
    --------
    float or array
        CDF value(s)
    """
    if np.any(x < 0):
        return np.where(x < 0, 0, 1 - np.exp(-(x / a) ** b))
    return 1 - np.exp(-(x / a) ** b)


def weibull_pdf(x, a, b):
    """
    Weibull distribution PDF: f(x|a, b) = (b/a^b) * x^(b-1) * exp(-(x/a)^b)
    
    Parameters:
    -----------
    x : float or array-like
        Value(s) at which to evaluate the PDF
    a : float
        Scale parameter (a > 0)
    b : float
        Shape parameter (b > 0)
    
    Returns:
    --------
    float or array
        PDF value(s)
    """
    x = np.asarray(x)
    if np.any(x < 0):
        return np.where(x < 0, 0, (b / a**b) * x**(b-1) * np.exp(-(x / a)**b))
    return (b / a**b) * x**(b-1) * np.exp(-(x / a)**b)


# Example usage and testing
if __name__ == "__main__":
    # Parameters from the table (Distribution in minutes)
    # Loglogistic uses Logistic distribution on ln(time) parameterization
    params = {
        'Gamma': {'a': 1.291, 'b': 1.732},
        'Lognormal': {'a': 0.492, 'b': 0.967},
        'Loglogistic': {'a': 0.498, 'b': 0.587},  # a is location on log scale, b is scale on log scale
        'Weibull': {'a': 2.321, 'b': 1.195}
    }
    
    # Create x values for plotting (in minutes) - extended to 8 minutes like reference
    x = np.linspace(0.01, 8, 1000)
    
    # Calculate CDF values for all distributions
    y_gamma_cdf = gamma_cdf(x, params['Gamma']['a'], params['Gamma']['b'])
    y_lognormal_cdf = lognormal_cdf(x, params['Lognormal']['a'], params['Lognormal']['b'])
    y_loglogistic_cdf = loglogistic_cdf(x, params['Loglogistic']['a'], params['Loglogistic']['b'])
    y_weibull_cdf = weibull_cdf(x, params['Weibull']['a'], params['Weibull']['b'])
    
    # Calculate PDF values for all distributions
    y_gamma_pdf = gamma_pdf(x, params['Gamma']['a'], params['Gamma']['b'])
    y_lognormal_pdf = lognormal_pdf(x, params['Lognormal']['a'], params['Lognormal']['b'])
    y_loglogistic_pdf = loglogistic_pdf(x, params['Loglogistic']['a'], params['Loglogistic']['b'])
    y_weibull_pdf = weibull_pdf(x, params['Weibull']['a'], params['Weibull']['b'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # CDF Plot
    ax1.plot(x, y_gamma_cdf, 'b-', linewidth=2.5, label=f"Gamma (a={params['Gamma']['a']}, b={params['Gamma']['b']})")
    ax1.plot(x, y_lognormal_cdf, 'r-', linewidth=2.5, label=f"Lognormal (a={params['Lognormal']['a']}, b={params['Lognormal']['b']})")
    ax1.plot(x, y_loglogistic_cdf, 'g-', linewidth=2.5, label=f"Loglogistic (a={params['Loglogistic']['a']}, b={params['Loglogistic']['b']})")
    ax1.plot(x, y_weibull_cdf, 'm-', linewidth=2.5, label=f"Weibull (a={params['Weibull']['a']}, b={params['Weibull']['b']})")
    ax1.set_title('Cumulative Distribution Functions (CDF)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 8])
    ax1.set_ylim([0, 1])
    
    # PDF Plot
    ax2.plot(x, y_gamma_pdf, 'b-', linewidth=2.5, label=f"Gamma (a={params['Gamma']['a']}, b={params['Gamma']['b']})")
    ax2.plot(x, y_lognormal_pdf, 'r-', linewidth=2.5, label=f"Lognormal (a={params['Lognormal']['a']}, b={params['Lognormal']['b']})")
    ax2.plot(x, y_loglogistic_pdf, 'g-', linewidth=2.5, label=f"Loglogistic (a={params['Loglogistic']['a']}, b={params['Loglogistic']['b']})")
    ax2.plot(x, y_weibull_pdf, 'm-', linewidth=2.5, label=f"Weibull (a={params['Weibull']['a']}, b={params['Weibull']['b']})")
    ax2.set_title('Probability Density Functions (PDF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 8])
    
    plt.tight_layout()
    plt.savefig('distributions_cdf_pdf.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'distributions_cdf_pdf.png'")
    plt.show()
    
    # Print some sample values
    print("\n" + "=" * 60)
    print("Sample CDF Values at Selected Time Points")
    print("=" * 60)
    test_times = np.array([0.5, 1.0, 2.0, 5.0])
    
    print(f"\nTime (minutes): {test_times}")
    print(f"\nGamma CDF:       {gamma_cdf(test_times, params['Gamma']['a'], params['Gamma']['b'])}")
    print(f"Lognormal CDF:   {lognormal_cdf(test_times, params['Lognormal']['a'], params['Lognormal']['b'])}")
    print(f"Loglogistic CDF: {loglogistic_cdf(test_times, params['Loglogistic']['a'], params['Loglogistic']['b'])}")
    print(f"Weibull CDF:     {weibull_cdf(test_times, params['Weibull']['a'], params['Weibull']['b'])}")
    
    print(f"\nGamma PDF:       {gamma_pdf(test_times, params['Gamma']['a'], params['Gamma']['b'])}")
    print(f"Lognormal PDF:   {lognormal_pdf(test_times, params['Lognormal']['a'], params['Lognormal']['b'])}")
    print(f"Loglogistic PDF: {loglogistic_pdf(test_times, params['Loglogistic']['a'], params['Loglogistic']['b'])}")
    print(f"Weibull PDF:     {weibull_pdf(test_times, params['Weibull']['a'], params['Weibull']['b'])}")
    print("\n" + "=" * 60)
