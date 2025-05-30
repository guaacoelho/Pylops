
class PhysicalPropertiesMixin:
    """
    Mixin that provides access to various physical properties such as 'vs',
    'rho', 'lambda', etc., typically used in wave propagation or
    geophysical modeling contexts.

    This mixin adds property accessors (e.g., 'self.vs', 'self.rho', etc.)
    that internally call the `_get_property(attr_name)` method to retrieve
    and process model parameters.

    Requirements:
    -------------
    The class that inherits from this mixin must define the following:

    - `self.model`: An object that exposes the required physical attributes
       as attributes.

    - `self._crop_model(data, nbl)`: A method that takes in the raw data
       and the model's boundary size (`nbl`) and returns the appropriately
       cropped data.
    """
    def _get_property(self, attr_name):
        attr = getattr(self.model, attr_name, None)
        if attr is not None:
            return self._crop_model(attr.data, self.model.nbl)
        raise AttributeError(
            f"{type(self).__name__} doesn't have parameter {attr_name}")

    @property
    def vp(self):
        return self._get_property("vp")

    @property
    def vs(self):
        return self._get_property("vs")

    @property
    def rho(self):
        return self._get_property("rho")

    @property
    def lam(self):
        return self._get_property("lam")

    @property
    def mu(self):
        return self._get_property("mu")

    @property
    def Ip(self):
        return self._get_property("Ip")

    @property
    def Is(self):
        return self._get_property("Is")

    @property
    def delta(self):
        return self._get_property("delta")

    @property
    def epsilon(self):
        return self._get_property("epsilon")

    @property
    def phi(self):
        return self._get_property("phi")

    @property
    def gamma(self):
        return self._get_property("gamma")

    @property
    def C11(self):
        return self._get_property("C11")

    @property
    def C22(self):
        return self._get_property("C22")

    @property
    def C33(self):
        return self._get_property("C33")

    @property
    def C44(self):
        return self._get_property("C44")

    @property
    def C55(self):
        return self._get_property("C55")

    @property
    def C66(self):
        return self._get_property("C66")

    @property
    def C12(self):
        return self._get_property("C12")

    @property
    def C21(self):
        return self._get_property("C21")

    @property
    def C13(self):
        return self._get_property("C13")

    @property
    def C31(self):
        return self._get_property("C31")

    @property
    def C23(self):
        return self._get_property("C23")

    @property
    def C32(self):
        return self._get_property("C32")
