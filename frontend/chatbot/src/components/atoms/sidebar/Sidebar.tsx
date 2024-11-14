import React, { useEffect, useRef } from "react";
import SEMBOT_LOGO from "@assets/images/logo.png";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";

type ButtonWithIconProps = React.ComponentProps<typeof ButtonWithIcon>;

export interface SidebarProps {
  components: ButtonWithIconProps[];
  footerComponents: ButtonWithIconProps[];
  isRule: boolean;
  hasMore?: boolean;
  onLoadMore?: () => void;
  isLoading?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  components = [],
  footerComponents = [],
  isRule,
  hasMore = false,
  onLoadMore = () => {},
  isLoading = false,
}) => {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const lastItemRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const firstEntry = entries[0];
        if (firstEntry.isIntersecting && hasMore && !isLoading) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (lastItemRef.current) {
      observerRef.current.observe(lastItemRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, isLoading, onLoadMore]);

  return (
    <div>
      <div className="w-full border-r border-indigo-900 bg-semesBlue">
        <div className="flex h-screen flex-col justify-between pt-2 pb-6">
          <div>
            {/* Logo Header */}
            <div className="w-full flex flex-row mb-2">
              <img src={SEMBOT_LOGO} alt="SEMBOT LOGO" />
              <div className="text-white text-xl mt-2 ml-1">SEMBOT</div>
            </div>

            {/* Scrollable Component List */}
            <div
              className="w-full flex flex-col space-y-2 overflow-y-auto max-h-[calc(100vh-250px)]
                        [&::-webkit-scrollbar]:w-0"
            >
              {components.map((buttonProps, index) => (
                <React.Fragment key={index}>
                  <ButtonWithIcon {...buttonProps} />
                  {index === 0 && isRule && (
                    <div className="text-white text-xl">규정 즐겨찾기</div>
                  )}
                </React.Fragment>
              ))}

              {/* Loading indicator and sentinel */}
              <div ref={lastItemRef}>
                {isLoading && (
                  <div className="text-white text-center py-2">Loading...</div>
                )}
              </div>
            </div>
          </div>

          {/* Footer Section */}
          <div>
            <hr className="mx-2 mb-3 border border-gray-100" />
            <div className="w-full mb-3 flex flex-col space-y-2">
              {footerComponents.map((buttonProps, index) => (
                <ButtonWithIcon key={index} {...buttonProps} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
